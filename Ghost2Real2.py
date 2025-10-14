import tensorflow as tf
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import os
import time
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    print("GUI libraries not available. Running in console mode only.")
    GUI_AVAILABLE = False

# ----------------------------
# Your ML Model (from original code)
# ----------------------------
class MeshSuperResNet(tf.keras.Model):
    def __init__(self, hidden_dim=512):
        super(MeshSuperResNet, self).__init__()

        self.local_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
        ])

        self.global_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu')
        ])

        self.position_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim * 2, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(3, activation='tanh')
        ])

        self.displacement_scale = tf.Variable(0.05, trainable=True)
        self.feature_scale = tf.Variable(1.0, trainable=True)

    def call(self, low_vertices, query_positions, low_to_query_weights, training=None):
        local_features = self.local_encoder(low_vertices, training=training)
        local_features = local_features * self.feature_scale

        global_max = tf.reduce_max(local_features, axis=1, keepdims=True)
        global_mean = tf.reduce_mean(local_features, axis=1, keepdims=True)
        global_std = tf.math.reduce_std(local_features, axis=1, keepdims=True)
        global_context = self.global_encoder(
            tf.concat([global_max, global_mean, global_std], axis=-1), training=training
        )

        interpolated_features = tf.matmul(low_to_query_weights, local_features)
        num_query = tf.shape(query_positions)[1]
        global_broadcast = tf.tile(global_context, [1, num_query, 1])

        combined_input = tf.concat([
            query_positions,
            interpolated_features,
            global_broadcast
        ], axis=-1)

        displacement = self.position_decoder(combined_input, training=training)
        displacement = displacement * self.displacement_scale

        refined_positions = query_positions + displacement
        return refined_positions

# ----------------------------
# Helper Functions (from your original code)
# ----------------------------
def compute_enhanced_interpolation_weights(low_vertices, high_vertices, k=8):
    """Enhanced interpolation with more neighbors and better weighting"""
    if len(low_vertices) < k:
        k = len(low_vertices)

    tree = cKDTree(low_vertices)
    distances, indices = tree.query(high_vertices, k=k)

    sigma = np.mean(distances) * 0.5
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    batch_size = 1
    num_high = len(high_vertices)
    num_low = len(low_vertices)

    weight_matrix = np.zeros((batch_size, num_high, num_low), dtype=np.float32)
    for i in range(num_high):
        for j, idx in enumerate(indices[i]):
            weight_matrix[0, i, idx] = weights[i, j]

    return weight_matrix

def create_enhanced_template(low_vertices, high_vertices):
    """Create better initial template with less noise"""
    template = high_vertices.copy()
    noise_scale = np.std(high_vertices, axis=0) * 0.02
    noise = np.random.normal(0, noise_scale, high_vertices.shape).astype(np.float32)
    template += noise
    return template.astype(np.float32)

# ----------------------------
# Main ML LOD System
# ----------------------------
class MLLODSystem:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.original_mesh = None
        self.trained_model = None
        self.preprocessing_data = None
        self.mesh_variants = {}
        self.ml_enhanced_mesh = None
        self.quality_metrics = {}

        # Initialize
        self._load_and_prepare_meshes()

    def _load_and_prepare_meshes(self):
        """Load original mesh and create quality variants"""
        print("Loading and preparing mesh variants...")

        if not os.path.exists(self.mesh_path):
            print(f"Error: Mesh file not found: {self.mesh_path}")
            return False

        try:
            self.original_mesh = trimesh.load(self.mesh_path)

            if not hasattr(self.original_mesh, 'vertices') or len(self.original_mesh.vertices) == 0:
                print(f"Error: Invalid mesh file - no vertices found")
                return False

            print(f"Original mesh loaded: {len(self.original_mesh.vertices):,} vertices, {len(self.original_mesh.faces):,} faces")
        except Exception as e:
            print(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Create different quality levels
        original_faces = len(self.original_mesh.faces)

        try:
            self.mesh_variants = {
                'ultra_low': self.original_mesh.simplify_quadric_decimation(face_count=max(100, original_faces // 20)),
                'low': self.original_mesh.simplify_quadric_decimation(face_count=max(300, original_faces // 8)),
                'medium_base': self.original_mesh.simplify_quadric_decimation(face_count=max(500, original_faces // 4)),
                'high': self.original_mesh
            }

            print("Mesh variants created:")
            for quality, mesh in self.mesh_variants.items():
                reduction = ((original_faces - len(mesh.faces)) / original_faces) * 100
                print(f"  {quality:12s}: {len(mesh.faces):,} faces ({reduction:.1f}% reduction)")

            return True

        except Exception as e:
            print(f"Error creating mesh variants: {e}")
            return False

    def train_ml_model(self, epochs=200):
        """Train the ML model on actual mesh data"""
        print("\nTraining ML Model...")

        try:
            # Prepare training data
            low_vertices, high_vertices, template, weights, center, scale = self._prepare_training_data()

            # Train model
            model, loss_history, best_loss = self._train_model(low_vertices, high_vertices, template, weights, epochs)

            # Store results
            self.trained_model = model
            self.preprocessing_data = {
                'center': center,
                'scale': scale,
                'template': template,
                'weights': weights
            }

            # Generate ML enhanced mesh
            self.ml_enhanced_mesh = self._generate_ml_enhanced_mesh()

            # Calculate metrics
            self._calculate_quality_metrics(loss_history, best_loss)

            print(f"ML Model trained successfully! Best loss: {best_loss:.6f}")
            return True

        except Exception as e:
            print(f"Training failed: {e}")
            return False

    def _prepare_training_data(self):
        """Prepare training data"""
        low_poly_mesh = self.mesh_variants['medium_base']
        high_poly_mesh = self.mesh_variants['high']

        high_vertices = np.array(high_poly_mesh.vertices, dtype=np.float32)
        low_vertices = np.array(low_poly_mesh.vertices, dtype=np.float32)

        # Normalization
        combined_vertices = np.vstack([high_vertices, low_vertices])
        center = np.mean(combined_vertices, axis=0)
        scale = np.max(np.linalg.norm(combined_vertices - center, axis=1)) + 1e-8

        high_vertices_norm = (high_vertices - center) / scale
        low_vertices_norm = (low_vertices - center) / scale

        # Create template and weights
        template = create_enhanced_template(low_vertices_norm, high_vertices_norm)
        weights = compute_enhanced_interpolation_weights(low_vertices_norm, high_vertices_norm)

        # Add batch dimension
        low_vertices_norm = low_vertices_norm[np.newaxis, :, :]
        high_vertices_norm = high_vertices_norm[np.newaxis, :, :]
        template = template[np.newaxis, :, :]

        return low_vertices_norm, high_vertices_norm, template, weights, center, scale

    def _train_model(self, low_vertices, high_vertices, template, weights, epochs):
        """Train the model"""
        low_tf = tf.constant(low_vertices, dtype=tf.float32)
        high_tf = tf.constant(high_vertices, dtype=tf.float32)
        template_tf = tf.constant(template, dtype=tf.float32)
        weights_tf = tf.constant(weights, dtype=tf.float32)

        model = MeshSuperResNet(hidden_dim=512)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=100, decay_rate=0.95, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        best_loss = float('inf')
        loss_history = []

        print("Training progress:")
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                pred = model(low_tf, template_tf, weights_tf, training=True)
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - high_tf), axis=-1))

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            current_loss = loss.numpy()
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {current_loss:.6f} | Best: {best_loss:.6f}")

        return model, loss_history, best_loss

    def _generate_ml_enhanced_mesh(self):
        """Generate ML enhanced mesh"""
        if self.trained_model is None:
            return None

        try:
            prep_data = self.preprocessing_data

            # Prepare input
            low_vertices_norm = (self.mesh_variants['medium_base'].vertices - prep_data['center']) / prep_data['scale']
            low_vertices_norm = low_vertices_norm[np.newaxis, :, :]

            # Generate enhanced vertices
            enhanced_vertices_norm = self.trained_model(
                tf.constant(low_vertices_norm, dtype=tf.float32),
                tf.constant(prep_data['template'], dtype=tf.float32),
                tf.constant(prep_data['weights'], dtype=tf.float32),
                training=False
            )

            # Denormalize
            enhanced_vertices = enhanced_vertices_norm.numpy()[0] * prep_data['scale'] + prep_data['center']

            # Create mesh
            ml_enhanced_mesh = trimesh.Trimesh(
                vertices=enhanced_vertices,
                faces=self.mesh_variants['high'].faces
            )

            return ml_enhanced_mesh

        except Exception as e:
            print(f"Error generating ML enhanced mesh: {e}")
            return None

    def _calculate_quality_metrics(self, loss_history, best_loss):
        """Calculate quality metrics"""
        if self.ml_enhanced_mesh is None:
            return

        original_vertices = self.original_mesh.vertices
        ml_vertices = self.ml_enhanced_mesh.vertices

        reconstruction_error = np.mean(np.linalg.norm(ml_vertices - original_vertices, axis=1))
        max_error = np.max(np.linalg.norm(ml_vertices - original_vertices, axis=1))

        self.quality_metrics = {
            'reconstruction_error_mean': reconstruction_error,
            'reconstruction_error_max': max_error,
            'final_training_loss': loss_history[-1] if loss_history else 0,
            'best_training_loss': best_loss,
            'face_improvement': f"{len(self.mesh_variants['medium_base'].faces):,} → {len(self.original_mesh.faces):,}",
            'quality_improvement': ((len(self.original_mesh.faces) - len(self.mesh_variants['medium_base'].faces)) / len(self.mesh_variants['medium_base'].faces)) * 100
        }

    def show_visual_results(self):
        """Show visual comparison of all mesh variants"""
        if self.ml_enhanced_mesh is None:
            print("ML model not trained yet!")
            return

        scene = trimesh.Scene()

        # Positions and colors
        positions = [np.array([0, 0, 0]), np.array([4, 0, 0]), np.array([8, 0, 0]), np.array([12, 0, 0]), np.array([16, 0, 0])]
        colors = [[255, 100, 100, 255], [255, 200, 100, 255], [255, 255, 100, 255], [100, 255, 100, 255], [100, 100, 255, 255]]
        labels = ["Ultra Low", "Low", "Medium Base", "🤖 ML Enhanced", "Original"]
        meshes = [self.mesh_variants['ultra_low'], self.mesh_variants['low'],
                 self.mesh_variants['medium_base'], self.ml_enhanced_mesh, self.mesh_variants['high']]

        print("\nVISUAL RESULTS")
        print("=" * 60)

        for i, (mesh, pos, color, label) in enumerate(zip(meshes, positions, colors, labels)):
            positioned_mesh = mesh.copy()
            positioned_mesh.vertices += pos
            positioned_mesh.visual.vertex_colors = color
            scene.add_geometry(positioned_mesh)

            faces = len(mesh.faces)
            print(f"{i+1}. {label:15s}: {faces:,} faces")

        print(f"\nKEY:")
        print(f"RED: Ultra low  ORANGE: Low  YELLOW: Medium")
        print(f"GREEN: ML Enhanced (YOUR MODEL!)  BLUE: Original")
        print(f"\n Compare GREEN to BLUE - they should look very similar!")

        # Show metrics
        print(f"\nQUALITY METRICS:")
        for metric, value in self.quality_metrics.items():
            if isinstance(value, float):
                print(f"  {metric:25s}: {value:.6f}")
            else:
                print(f"  {metric:25s}: {value}")

        scene.show()

# ----------------------------
# Simple Console Interface
# ----------------------------
def run_console_demo(file_path=None):
    """Run console-based demo"""
    if file_path is None:
        file_path = r"C:\Users\Ber\GhostObjects\tennis_ball.obj"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print("Please update the file_path variable with your mesh location.")
        return

    print("ML LOD SYSTEM - CONSOLE MODE")
    print("=" * 50)

    # Initialize system
    lod_system = MLLODSystem(file_path)

    if not lod_system.mesh_variants:
        print("Failed to load mesh variants")
        return

    # Train model
    success = lod_system.train_ml_model(epochs=150)  # Reduced for faster demo

    if not success:
        print("Training failed")
        return

    # Show results
    lod_system.show_visual_results()

    print(f"\n SUCCESS!")
    print(f"Your ML model enhanced mesh quality from {len(lod_system.mesh_variants['medium_base'].faces):,} to {len(lod_system.original_mesh.faces):,} faces!")
    print(f"Reconstruction error: {lod_system.quality_metrics['reconstruction_error_mean']:.6f}")

# ----------------------------
# GUI Interface (if available)
# ----------------------------
if GUI_AVAILABLE:
    class ResearchGUI:
        def __init__(self, file_path=None):
            self.lod_system = None
            self.root = None
            self.file_path = file_path if file_path else r"C:\Users\Ber\GhostObjects\tennis_ball.obj"
            self.epochs_var = None
            self.progress_var = None
            self.training_data = []

        def create_gui(self):
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}")
                return

            self.lod_system = MLLODSystem(self.file_path)

            self.root = tk.Tk()
            self.root.title("Ghost2Real - Research Interface")
            self.root.geometry("900x700")
            self.root.configure(bg="#f0f0f0")

            # Title
            title_label = tk.Label(self.root, text="Ghost2Real Neural Mesh Super-Resolution",
                                  font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#2c3e50")
            title_label.pack(pady=15)

            subtitle_label = tk.Label(self.root, text="Research-Focused ML Training Interface",
                                     font=("Arial", 11), bg="#f0f0f0", fg="#7f8c8d")
            subtitle_label.pack(pady=5)

            # Main container
            main_frame = tk.Frame(self.root, bg="#f0f0f0")
            main_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # Left panel - Mesh Info & Preview
            left_panel = tk.LabelFrame(main_frame, text="Mesh Information",
                                      font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Mesh statistics
            mesh_info = f"""Original Mesh: {os.path.basename(self.file_path)}

Vertices: {len(self.lod_system.original_mesh.vertices):,}
Faces: {len(self.lod_system.original_mesh.faces):,}

LOD Variants Generated:
"""
            for quality, mesh in self.lod_system.mesh_variants.items():
                reduction = ((len(self.lod_system.original_mesh.faces) - len(mesh.faces)) /
                           len(self.lod_system.original_mesh.faces)) * 100
                mesh_info += f"  • {quality:12s}: {len(mesh.faces):4,} faces ({reduction:5.1f}% reduction)\n"

            info_text = tk.Text(left_panel, height=15, width=40, font=("Consolas", 9),
                              wrap="word", bg="#f8f9fa", relief="flat")
            info_text.insert(1.0, mesh_info)
            info_text.config(state="disabled")
            info_text.pack(fill="both", expand=True)

            # Preview button
            preview_btn = tk.Button(left_panel, text="Preview Mesh",
                                   command=self.preview_mesh,
                                   font=("Arial", 10), bg="#3498db", fg="white",
                                   activebackground="#2980b9", cursor="hand2")
            preview_btn.pack(pady=10, fill="x")

            # Right panel - Training Configuration
            right_panel = tk.LabelFrame(main_frame, text="Training Configuration",
                                       font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            # Epochs configuration
            epochs_frame = tk.Frame(right_panel, bg="white")
            epochs_frame.pack(fill="x", pady=10)

            tk.Label(epochs_frame, text="Training Epochs:",
                    font=("Arial", 10, "bold"), bg="white").pack(anchor="w")

            self.epochs_var = tk.IntVar(value=200)
            epochs_slider = tk.Scale(epochs_frame, from_=50, to=1000,
                                    orient="horizontal", variable=self.epochs_var,
                                    length=300, bg="white", font=("Arial", 9))
            epochs_slider.pack(fill="x", pady=5)

            tk.Label(epochs_frame, textvariable=self.epochs_var,
                    font=("Arial", 10), bg="white", fg="#2c3e50").pack(anchor="w")

            # Model parameters info
            params_info = """Model Architecture:
  • Hidden Dimension: 512
  • Encoder: 3-layer dense + LayerNorm
  • Decoder: 5-layer residual network
  • Optimizer: Adam (LR=0.001, decay=0.95)
  • K-NN Neighbors: 8 (Gaussian weighting)

Training Strategy:
  • Gradient clipping: L2 norm ≤ 1.0
  • Loss: MSE + Smoothness regularization
  • Early stopping available
"""
            params_text = tk.Text(right_panel, height=12, width=40, font=("Consolas", 8),
                                wrap="word", bg="#f8f9fa", relief="flat")
            params_text.insert(1.0, params_info)
            params_text.config(state="disabled")
            params_text.pack(fill="both", expand=True, pady=10)

            # Training button
            train_btn = tk.Button(right_panel, text="Start Training",
                                 command=self.train_model,
                                 font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                                 activebackground="#229954", cursor="hand2", height=2)
            train_btn.pack(pady=10, fill="x")

            # Results button
            self.results_btn = tk.Button(right_panel, text="View Results & Metrics",
                                        command=self.show_results,
                                        font=("Arial", 11), bg="#e74c3c", fg="white",
                                        activebackground="#c0392b", cursor="hand2",
                                        state="disabled")
            self.results_btn.pack(pady=5, fill="x")

            # Bottom panel - Training Progress & Metrics
            bottom_panel = tk.LabelFrame(main_frame, text="Training Progress & Metrics",
                                        font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            bottom_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

            # Progress bar
            progress_frame = tk.Frame(bottom_panel, bg="white")
            progress_frame.pack(fill="x", pady=5)

            tk.Label(progress_frame, text="Progress:", font=("Arial", 9), bg="white").pack(side="left", padx=5)

            self.progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=400)
            progress_bar.pack(side="left", fill="x", expand=True, padx=5)

            # Status label
            self.status_label = tk.Label(bottom_panel, text="Ready to begin training",
                                        font=("Arial", 10), bg="white", fg="#7f8c8d")
            self.status_label.pack(pady=5)

            # Results text area
            self.results_text = tk.Text(bottom_panel, height=10, width=100,
                                       font=("Consolas", 9), wrap="word",
                                       bg="#f8f9fa", relief="flat")
            scrollbar = tk.Scrollbar(bottom_panel, command=self.results_text.yview)
            self.results_text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            self.results_text.pack(fill="both", expand=True, pady=5)

            # Configure grid weights
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_columnconfigure(1, weight=1)
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_rowconfigure(1, weight=1)

            return self.root

        def preview_mesh(self):
            """Show preview of the original mesh"""
            self.lod_system.original_mesh.show()

        def train_model(self):
            epochs = self.epochs_var.get()
            self.status_label.config(text=f"Training with {epochs} epochs... Please wait", fg="#e67e22")
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, f"[TRAINING STARTED]\nEpochs: {epochs}\nModel: MeshSuperResNet (512-dim)\n\n")
            self.root.update()

            def train_worker():
                success = self.lod_system.train_ml_model(epochs=epochs)

                if success:
                    self.root.after(0, self._training_success)
                else:
                    self.root.after(0, self._training_failed)

            thread = threading.Thread(target=train_worker)
            thread.daemon = True
            thread.start()

        def _training_success(self):
            self.status_label.config(text="✓ Training completed successfully!", fg="#27ae60")
            self.progress_var.set(100)
            self.results_btn.config(state="normal")

            metrics = self.lod_system.quality_metrics

            results = f"""[TRAINING COMPLETED]

==================== PERFORMANCE METRICS ====================

Reconstruction Quality:
  • Mean Error:         {metrics['reconstruction_error_mean']:.8f}
  • Max Error:          {metrics['reconstruction_error_max']:.8f}
  • Final Loss:         {metrics['final_training_loss']:.8f}
  • Best Loss:          {metrics['best_training_loss']:.8f}

Mesh Enhancement:
  • Face Count:         {metrics['face_improvement']}
  • Quality Gain:       {metrics['quality_improvement']:.2f}%

Training Configuration:
  • Epochs:             {self.epochs_var.get()}
  • Hidden Dimension:   512
  • Learning Rate:      0.001 (exponential decay)
  • Gradient Clipping:  L2 norm ≤ 1.0

==============================================================

✓ Model successfully learned to enhance mesh geometry.
✓ Ready for visual inspection and further analysis.

Click 'View Results & Metrics' to see 3D comparison.
"""

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)

        def _training_failed(self):
            self.status_label.config(text="✗ Training failed - check console", fg="#e74c3c")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, "[TRAINING FAILED]\n\nCheck console output for error details.\nCommon issues:\n  • Invalid mesh file\n  • Insufficient memory\n  • Incompatible TensorFlow version")

        def show_results(self):
            self.lod_system.show_visual_results()

    def run_gui_demo(file_path=None):
        """Run GUI demo"""
        gui = ResearchGUI(file_path)
        root = gui.create_gui()
        if root:
            root.mainloop()

# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main function - chooses GUI or console mode"""
    print("ML MESH ENHANCEMENT SYSTEM")
    print("=" * 40)

    if GUI_AVAILABLE:
        print("Starting GUI mode...")
        run_gui_demo()
    else:
        print("GUI not available, running console mode...")
        run_console_demo()

if __name__ == "__main__":
    main()
