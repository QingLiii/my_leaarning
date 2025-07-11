import tkinter as tk
import math
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

# Constants
WIDTH, HEIGHT = 800, 800
CENTER = np.array([WIDTH / 2, HEIGHT / 2])
HEPTAGON_RADIUS = 300
BALL_RADIUS = 15
GRAVITY = 0.3
FRICTION = 0.99
BALL_FRICTION = 0.98
BOUNCE_EFFICIENCY = 0.8
HEPTAGON_SPEED = 360 / 5  # degrees per second
FPS = 60
NUM_BALLS = 20

# Colors
COLORS = [
    "#f8b862", "#f6ad49", "#f39800", "#f08300", "#ec6d51", "#ee7948", "#ed6d3d",
    "#ec6800", "#ec6800", "#ee7800", "#eb6238", "#ea5506", "#ea5506", "#eb6101",
    "#e49e61", "#e45e32", "#e17b34", "#dd7a56", "#db8449", "#d66a35"
]

@dataclass
class Ball:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    color: str
    angle: float = 0.0  # rotation angle of the ball
    angular_velocity: float = 0.0

class SpinningHeptagonBalls:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()
        
        self.balls: List[Ball] = []
        self.heptagon_angle = 0.0  # current rotation angle of the heptagon
        
        self.create_balls()
        self.running = True
        
        self.root.bind("<Escape>", lambda e: self.quit())
        self.update()
        
    def create_balls(self):
        for i in range(NUM_BALLS):
            ball = Ball(
                id=i+1,
                position=np.array([CENTER[0], CENTER[1]], dtype=float),
                velocity=np.array([random.uniform(-2, 2), random.uniform(-2, 0)], dtype=float),
                radius=BALL_RADIUS,
                color=COLORS[i % len(COLORS)]
            )
            self.balls.append(ball)
    
    def get_heptagon_vertices(self, angle: float) -> List[Tuple[float, float]]:
        vertices = []
        for i in range(7):
            theta = math.radians(angle + i * (360 / 7))
            x = CENTER[0] + HEPTAGON_RADIUS * math.cos(theta)
            y = CENTER[1] + HEPTAGON_RADIUS * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        if projection < 0:
            return np.linalg.norm(point_vec)
        elif projection > line_len:
            return np.linalg.norm(point - line_end)
        perpendicular = point_vec - projection * line_unitvec
        return np.linalg.norm(perpendicular)
    
    def check_collision_with_heptagon(self, ball: Ball, vertices: List[Tuple[float, float]]) -> bool:
        for i in range(len(vertices)):
            start = np.array(vertices[i])
            end = np.array(vertices[(i + 1) % len(vertices)])
            distance = self.point_to_line_distance(ball.position, start, end)
            if distance <= ball.radius:
                # Calculate normal
                edge = end - start
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Reflect velocity
                dot_product = np.dot(ball.velocity, normal)
                ball.velocity = ball.velocity - 2 * dot_product * normal
                ball.velocity *= BOUNCE_EFFICIENCY
                
                # Move ball outside
                overlap = ball.radius - distance + 1
                ball.position += normal * overlap
                
                # Apply spin
                ball.angular_velocity += random.uniform(-0.5, 0.5)
                return True
        return False
    
    def check_ball_collisions(self):
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                ball1, ball2 = self.balls[i], self.balls[j]
                distance = np.linalg.norm(ball1.position - ball2.position)
                if distance < ball1.radius + ball2.radius:
                    # Collision detected
                    normal = (ball2.position - ball1.position) / distance
                    relative_velocity = ball2.velocity - ball1.velocity
                    speed = np.dot(relative_velocity, normal)
                    
                    if speed < 0:
                        continue
                    
                    # Calculate impulse
                    impulse = 2 * speed / (1/ball1.radius + 1/ball2.radius)
                    ball1.velocity += impulse * normal / ball1.radius
                    ball2.velocity -= impulse * normal / ball2.radius
                    
                    # Separate balls
                    overlap = (ball1.radius + ball2.radius) - distance
                    separation = normal * (overlap / 2)
                    ball1.position -= separation
                    ball2.position += separation
                    
                    # Apply spin
                    ball1.angular_velocity += random.uniform(-0.3, 0.3)
                    ball2.angular_velocity += random.uniform(-0.3, 0.3)
    
    def update_balls(self):
        # Update heptagon rotation
        self.heptagon_angle += HEPTAGON_SPEED / FPS
        
        vertices = self.get_heptagon_vertices(self.heptagon_angle)
        
        for ball in self.balls:
            # Apply gravity
            ball.velocity[1] += GRAVITY
            
            # Apply friction
            ball.velocity *= FRICTION
            
            # Update rotation
            ball.angle += ball.angular_velocity
            ball.angular_velocity *= BALL_FRICTION
            
            # Update position
            ball.position += ball.velocity
            
            # Check collisions
            self.check_collision_with_heptagon(ball, vertices)
        
        self.check_ball_collisions()
    
    def draw(self):
        self.canvas.delete("all")
        
        # Draw heptagon
        vertices = self.get_heptagon_vertices(self.heptagon_angle)
        self.canvas.create_polygon(
            *[(x, y) for x, y in vertices],
            outline='black',
            fill='',
            width=3
        )
        
        # Draw balls
        for ball in self.balls:
            x, y = ball.position
            self.canvas.create_oval(
                x - ball.radius, y - ball.radius,
                x + ball.radius, y + ball.radius,
                fill=ball.color,
                outline='black'
            )
            
            # Draw number
            self.canvas.create_text(
                x, y,
                text=str(ball.id),
                fill='white',
                font=('Arial', 10, 'bold')
            )
            
            # Draw rotation indicator (line from center to edge showing spin)
            indicator_length = ball.radius * 0.8
            indicator_x = x + indicator_length * math.cos(ball.angle)
            indicator_y = y + indicator_length * math.sin(ball.angle)
            self.canvas.create_line(x, y, indicator_x, indicator_y, fill='black', width=2)
    
    def update(self):
        if self.running:
            self.update_balls()
            self.draw()
            self.root.after(1000 // FPS, self.update)
    
    def quit(self):
        self.running = False
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spinning Heptagon Bouncing Balls")
    app = SpinningHeptagonBalls(root)
    root.mainloop()