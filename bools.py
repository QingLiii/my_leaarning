import tkinter as tk
import numpy as np
import math
import sys
from dataclasses import dataclass
from typing import List

# --- Constants ---

# Window and Canvas Configuration
WIDTH, HEIGHT = 800, 800
CANVAS_BG_COLOR = '#1a1a1a'
UPDATE_DELAY_MS = 10  # Corresponds to ~100 FPS, but actual FPS will be lower

# Simulation Physics Parameters
DT = 0.04  # Time step for physics integration, adjusted for smooth simulation
GRAVITY = np.array([0.0, 400.0])  # Gravity vector in pixels/s^2
HEPTAGON_ROT_SPEED = (2 * math.pi) / 5.0  # rad/s (360 degrees in 5 seconds)

# Heptagon Properties
HEPTAGON_CENTER = np.array([WIDTH / 2.0, HEIGHT / 2.0])
HEPTAGON_RADIUS = 360  # Distance from center to vertex
HEPTAGON_LINE_COLOR = '#ffffff'
HEPTAGON_LINE_WIDTH = 3

# Ball Properties
NUM_BALLS = 20
BALL_RADIUS = 25
BALL_MASS = 1.0
BALL_INERTIA = (2/5) * BALL_MASS * (BALL_RADIUS**2) # Moment of inertia for a solid sphere
BALL_WALL_RESTITUTION = 0.75  # Bounciness off walls
BALL_BALL_RESTITUTION = 0.9   # Bounciness between balls
WALL_FRICTION_COEFF = 0.2     # Frictional coefficient with heptagon walls
BALL_FRICTION_COEFF = 0.1     # Frictional coefficient between balls

# Colors for the 20 balls as specified
COLORS = [
    '#f8b862', '#f6ad49', '#f39800', '#f08300', '#ec6d51', 
    '#ee7948', '#ed6d3d', '#ec6800', '#ec6800', '#ee7800', 
    '#eb6238', '#ea5506', '#ea5506', '#eb6101', '#e49e61', 
    '#e45e32', '#e17b34', '#dd7a56', '#db8449', '#d66a35'
]

# --- Data Class for Ball State ---

@dataclass
class Ball:
    """Holds all state and properties for a single ball."""
    # Physical properties
    number: int
    radius: float
    mass: float
    inertia: float
    color: str
    
    # State variables
    position: np.ndarray
    velocity: np.ndarray
    angle: float = 0.0
    angular_velocity: float = 0.0
    
    # Tkinter canvas object IDs
    canvas_id: int = None
    text_id: int = None

# --- Main Simulation Class ---

class HeptagonBounceSimulation:
    """Manages the simulation, including Tkinter setup, physics, and rendering."""

    def __init__(self, root: tk.Tk):
        """Initializes the simulation environment."""
        self.root = root
        self.root.title("Bouncing Balls in a Spinning Heptagon")
        
        # Set up the canvas
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=CANVAS_BG_COLOR)
        self.canvas.pack()

        # Initialize simulation state
        self.heptagon_angle = 0.0
        self.heptagon_vertices = self._calculate_heptagon_vertices()
        self.heptagon_canvas_id = None
        
        self.balls: List[Ball] = self._create_balls()
        
        # Create canvas objects for the first time
        self._draw_heptagon()
        self._create_ball_canvas_objects()

        # Start the main simulation loop
        self.animate()

    def _create_balls(self) -> List[Ball]:
        """Creates the initial list of Ball objects."""
        balls = []
        for i in range(NUM_BALLS):
            # Start all balls at the center. A tiny random offset prevents perfect symmetry issues.
            offset = (np.random.rand(2) - 0.5) * 0.1
            ball = Ball(
                number=i + 1,
                radius=BALL_RADIUS,
                mass=BALL_MASS,
                inertia=BALL_INERTIA,
                color=COLORS[i % len(COLORS)],
                position=HEPTAGON_CENTER + offset,
                velocity=np.array([0.0, 0.0])
            )
            balls.append(ball)
        return balls

    def _calculate_heptagon_vertices(self) -> List[np.ndarray]:
        """Calculates the 7 vertex coordinates of the heptagon based on its current angle."""
        vertices = []
        for i in range(7):
            angle = (2 * math.pi * i / 7) + self.heptagon_angle
            x = HEPTAGON_CENTER[0] + HEPTAGON_RADIUS * math.cos(angle)
            y = HEPTAGON_CENTER[1] + HEPTAGON_RADIUS * math.sin(angle)
            vertices.append(np.array([x, y]))
        return vertices

    def _update_physics(self):
        """Applies physics rules to all balls for one time step."""
        # Update heptagon rotation
        self.heptagon_angle += HEPTAGON_ROT_SPEED * DT
        self.heptagon_vertices = self._calculate_heptagon_vertices()

        # Update each ball's state
        for ball in self.balls:
            # Apply gravity
            ball.velocity += GRAVITY * DT
            # Update position and rotation
            ball.position += ball.velocity * DT
            ball.angle += ball.angular_velocity * DT

        # Handle collisions with multiple passes for stability
        for _ in range(3): # Iterative solver for collisions
            # Ball-to-wall collisions
            for ball in self.balls:
                self._handle_wall_collisions(ball)
            
            # Ball-to-ball collisions
            for i in range(len(self.balls)):
                for j in range(i + 1, len(self.balls)):
                    self._handle_ball_collisions(self.balls[i], self.balls[j])

    def _handle_wall_collisions(self, ball: Ball):
        """Detects and resolves collisions between a ball and the heptagon walls."""
        for i in range(7):
            p1 = self.heptagon_vertices[i]
            p2 = self.heptagon_vertices[(i + 1) % 7]
            
            wall_vec = p2 - p1
            point_vec = ball.position - p1
            
            # Project ball's position onto the wall vector to find the closest point
            t = np.dot(point_vec, wall_vec) / np.dot(wall_vec, wall_vec)
            t = np.clip(t, 0, 1) # Clamp to the line segment
            
            closest_point = p1 + t * wall_vec
            collision_vec = ball.position - closest_point
            distance = np.linalg.norm(collision_vec)

            if distance < ball.radius:
                # --- Collision Detected ---
                
                # 1. Resolve Overlap
                overlap = ball.radius - distance
                normal = collision_vec / distance
                ball.position += normal * overlap

                # 2. Calculate Collision Response (Bounce)
                # Velocity of the wall at the collision point
                wall_point_vec = closest_point - HEPTAGON_CENTER
                wall_velocity = np.array([-wall_point_vec[1], wall_point_vec[0]]) * HEPTAGON_ROT_SPEED
                
                relative_velocity = ball.velocity - wall_velocity
                vel_along_normal = np.dot(relative_velocity, normal)

                # Only apply impulse if ball is moving towards the wall
                if vel_along_normal < 0:
                    # Calculate impulse for bounce
                    j = -(1 + BALL_WALL_RESTITUTION) * vel_along_normal
                    impulse = j * normal
                    ball.velocity += impulse # Mass is 1.0

                    # 3. Apply Friction
                    tangent = np.array([-normal[1], normal[0]])
                    vel_along_tangent = np.dot(relative_velocity, tangent)
                    
                    # Friction impulse (dynamic friction)
                    friction_impulse_magnitude = -vel_along_tangent * WALL_FRICTION_COEFF
                    friction_impulse = np.clip(friction_impulse_magnitude, -abs(j * WALL_FRICTION_COEFF), abs(j * WALL_FRICTION_COEFF))
                    
                    ball.velocity += friction_impulse * tangent

                    # 4. Apply Torque from friction
                    torque = -friction_impulse * ball.radius
                    ball.angular_velocity += torque / ball.inertia


    def _handle_ball_collisions(self, b1: Ball, b2: Ball):
        """Detects and resolves collisions between two balls."""
        collision_vec = b1.position - b2.position
        distance = np.linalg.norm(collision_vec)

        if distance < b1.radius + b2.radius:
            # --- Collision Detected ---
            
            # 1. Resolve Overlap
            normal = collision_vec / distance
            overlap = (b1.radius + b2.radius) - distance
            b1.position += normal * overlap / 2
            b2.position -= normal * overlap / 2

            # 2. Calculate Collision Response (Bounce)
            relative_velocity = b1.velocity - b2.velocity
            vel_along_normal = np.dot(relative_velocity, normal)
            
            # Only apply impulse if balls are moving towards each other
            if vel_along_normal < 0:
                j = -(1 + BALL_BALL_RESTITUTION) * vel_along_normal
                j /= (1/b1.mass + 1/b2.mass) # Both masses are 1.0, so this is j/2
                
                impulse = j * normal
                b1.velocity += impulse / b1.mass
                b2.velocity -= impulse / b2.mass

                # 3. Apply Friction (simplified model)
                tangent = np.array([-normal[1], normal[0]])
                vel_along_tangent = np.dot(relative_velocity, tangent)
                
                friction_impulse_magnitude = -vel_along_tangent * BALL_FRICTION_COEFF
                friction_impulse = np.clip(friction_impulse_magnitude, -abs(j * BALL_FRICTION_COEFF), abs(j * BALL_FRICTION_COEFF))
                
                # Apply tangential impulse
                b1.velocity += friction_impulse * tangent / b1.mass
                b2.velocity -= friction_impulse * tangent / b2.mass
                
                # 4. Apply Torque from friction
                torque = -friction_impulse * b1.radius
                b1.angular_velocity += torque / b1.inertia
                b2.angular_velocity -= torque / b2.inertia


    def _draw_heptagon(self):
        """Draws or moves the heptagon on the canvas."""
        flat_coords = [coord for vertex in self.heptagon_vertices for coord in vertex]
        if self.heptagon_canvas_id:
            self.canvas.coords(self.heptagon_canvas_id, flat_coords)
        else:
            self.heptagon_canvas_id = self.canvas.create_polygon(
                flat_coords, 
                fill='', 
                outline=HEPTAGON_LINE_COLOR, 
                width=HEPTAGON_LINE_WIDTH
            )

    def _create_ball_canvas_objects(self):
        """Creates the initial circle and text objects for each ball."""
        for ball in self.balls:
            x0, y0 = ball.position - ball.radius
            x1, y1 = ball.position + ball.radius
            ball.canvas_id = self.canvas.create_oval(x0, y0, x1, y1, fill=ball.color, outline='')
            ball.text_id = self.canvas.create_text(
                ball.position[0], ball.position[1],
                text=str(ball.number),
                font=('Helvetica', '12', 'bold'),
                fill='black'
            )

    def _update_graphics(self):
        """Updates the positions and rotations of all objects on the canvas."""
        self._draw_heptagon()
        for ball in self.balls:
            x0, y0 = ball.position - ball.radius
            x1, y1 = ball.position + ball.radius
            self.canvas.coords(ball.canvas_id, x0, y0, x1, y1)
            
            # Update text position and rotation
            self.canvas.coords(ball.text_id, ball.position[0], ball.position[1])
            # Convert angle from radians to degrees for tkinter
            self.canvas.itemconfig(ball.text_id, angle=-math.degrees(ball.angle))

    def animate(self):
        """The main loop of the simulation."""
        try:
            self._update_physics()
            self._update_graphics()
            self.root.after(UPDATE_DELAY_MS, self.animate)
        except tk.TclError:
            # Handle the case where the window is closed
            print("Simulation window closed.")
            sys.exit(0)

# --- Main Execution ---

if __name__ == "__main__":
    main_window = tk.Tk()
    simulation = HeptagonBounceSimulation(main_window)
    main_window.mainloop()
