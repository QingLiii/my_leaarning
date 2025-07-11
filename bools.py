import tkinter as tk
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Constants
NUM_BALLS = 20
BALL_RADIUS = 15
HEPTAGON_RADIUS = 300  # Distance from center to vertex
GRAVITY = 0.5
FRICTION = 0.98
ANGULAR_FRICTION = 0.99
BOUNCE_DAMPING = 0.8
COLLISION_DAMPING = 0.9
HEPTAGON_ROTATION_SPEED = 360 / 5  # degrees per second
FPS = 60

# Colors
COLORS = [
    "#f8b862", "#f6ad49", "#f39800", "#f08300", "#ec6d51",
    "#ee7948", "#ed6d3d", "#ec6800", "#ec6800", "#ee7800",
    "#eb6238", "#ea5506", "#ea5506", "#eb6101", "#e49e61",
    "#e45e32", "#e17b34", "#dd7a56", "#db8449", "#d66a35"
]

@dataclass
class Vector2:
    x: float
    y: float
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        l = self.length()
        if l == 0:
            return Vector2(0, 0)
        return Vector2(self.x/l, self.y/l)
    
    def rotate(self, angle):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return Vector2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

class Ball:
    def __init__(self, number: int, color: str, position: Vector2):
        self.number = number
        self.color = color
        self.position = position
        self.velocity = Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.angular_velocity = random.uniform(-5, 5)
        self.rotation = 0
    
    def update(self, dt: float):
        # Apply gravity
        self.velocity.y += GRAVITY * dt
        
        # Apply friction
        self.velocity = self.velocity * FRICTION
        
        # Update rotation
        self.angular_velocity *= ANGULAR_FRICTION
        self.rotation += self.angular_velocity * dt
        
        # Update position
        self.position = self.position + self.velocity * dt
    
    def draw(self, canvas: tk.Canvas):
        x, y = self.position.x, self.position.y
        canvas.create_oval(
            x - BALL_RADIUS, y - BALL_RADIUS,
            x + BALL_RADIUS, y + BALL_RADIUS,
            fill=self.color, outline="black"
        )
        
        # Draw number
        canvas.create_text(
            x, y,
            text=str(self.number),
            font=("Arial", 12, "bold"),
            fill="white"
        )
        
        # Draw rotation indicator (a small line from center to edge)
        indicator = Vector2(BALL_RADIUS - 5, 0).rotate(self.rotation)
        canvas.create_line(
            x, y,
            x + indicator.x, y + indicator.y,
            fill="white", width=2
        )

class Heptagon:
    def __init__(self, center: Vector2, radius: float):
        self.center = center
        self.radius = radius
        self.rotation = 0
    
    def get_vertices(self) -> List[Vector2]:
        vertices = []
        for i in range(7):
            angle = math.radians(360/7 * i + self.rotation)
            x = self.center.x + self.radius * math.cos(angle)
            y = self.center.y + self.radius * math.sin(angle)
            vertices.append(Vector2(x, y))
        return vertices
    
    def get_edges(self) -> List[Tuple[Vector2, Vector2]]:
        vertices = self.get_vertices()
        edges = []
        for i in range(7):
            edges.append((vertices[i], vertices[(i+1)%7]))
        return edges
    
    def get_normal(self, edge: Tuple[Vector2, Vector2]) -> Vector2:
        p1, p2 = edge
        edge_vec = p2 - p1
        normal = Vector2(-edge_vec.y, edge_vec.x).normalize()
        return normal
    
    def check_collision(self, ball: Ball) -> Tuple[bool, Vector2, Vector2]:
        """Check collision with heptagon walls. Returns (collision, normal, contact_point)"""
        for edge in self.get_edges():
            p1, p2 = edge
            edge_vec = p2 - p1
            edge_len = edge_vec.length()
            edge_dir = edge_vec.normalize()
            
            # Vector from edge start to ball center
            to_ball = ball.position - p1
            
            # Projection of to_ball onto edge
            projection = to_ball.dot(edge_dir)
            projection = max(0, min(projection, edge_len))
            
            # Closest point on edge
            closest_point = p1 + edge_dir * projection
            
            # Distance from ball to edge
            distance = (ball.position - closest_point).length()
            
            if distance <= BALL_RADIUS:
                normal = (ball.position - closest_point).normalize()
                return (True, normal, closest_point)
        
        return (False, Vector2(0, 0), Vector2(0, 0))
    
    def update(self, dt: float):
        self.rotation += HEPTAGON_ROTATION_SPEED * dt
    
    def draw(self, canvas: tk.Canvas):
        vertices = self.get_vertices()
        points = []
        for v in vertices:
            points.extend([v.x, v.y])
        
        canvas.create_polygon(points, fill="", outline="black", width=3)

class Simulation:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()
        
        self.center = Vector2(400, 300)
        self.heptagon = Heptagon(self.center, HEPTAGON_RADIUS)
        
        self.balls = []
        for i in range(NUM_BALLS):
            ball = Ball(
                i+1,
                COLORS[i],
                Vector2(self.center.x, self.center.y)
            )
            self.balls.append(ball)
        
        self.last_time = 0
        self.running = True
        
        self.root.after(1000//FPS, self.update)
    
    def check_ball_collision(self, ball1: Ball, ball2: Ball) -> bool:
        distance = (ball1.position - ball2.position).length()
        return distance < 2 * BALL_RADIUS
    
    def resolve_ball_collision(self, ball1: Ball, ball2: Ball):
        # Vector between centers
        delta = ball2.position - ball1.position
        distance = delta.length()
        
        if distance == 0:
            return
        
        # Normal direction
        normal = delta.normalize()
        
        # Separate balls
        overlap = 2 * BALL_RADIUS - distance
        separation = normal * (overlap / 2)
        ball1.position = ball1.position - separation
        ball2.position = ball2.position + separation
        
        # Relative velocity
        v_rel = ball2.velocity - ball1.velocity
        
        # Only resolve if balls are moving towards each other
        if v_rel.dot(normal) > 0:
            return
        
        # Calculate impulse
        impulse = 2 * v_rel.dot(normal) / 2  # Assuming equal mass
        
        # Apply impulse
        ball1.velocity = ball1.velocity + normal * impulse * COLLISION_DAMPING
        ball2.velocity = ball2.velocity - normal * impulse * COLLISION_DAMPING
        
        # Transfer some angular momentum
        avg_angular = (ball1.angular_velocity + ball2.angular_velocity) / 2
        ball1.angular_velocity = avg_angular * ANGULAR_FRICTION
        ball2.angular_velocity = avg_angular * ANGULAR_FRICTION
    
    def update(self):
        current_time = self.root.tk.call('after', 'info')
        if self.last_time == 0:
            self.last_time = current_time
            self.root.after(1000//FPS, self.update)
            return
        
        dt = 1/FPS
        
        # Update heptagon
        self.heptagon.update(dt)
        
        # Update balls
        for ball in self.balls:
            ball.update(dt)
            
            # Check collision with heptagon
            collision, normal, contact_point = self.heptagon.check_collision(ball)
            if collision:
                # Reflect velocity
                dot_product = ball.velocity.dot(normal)
                ball.velocity = ball.velocity - normal * (2 * dot_product) * BOUNCE_DAMPING
                
                # Limit bounce height (not exceeding heptagon radius)
                max_bounce_speed = math.sqrt(2 * GRAVITY * (HEPTAGON_RADIUS - BALL_RADIUS))
                speed = ball.velocity.length()
                if speed > max_bounce_speed:
                    ball.velocity = ball.velocity.normalize() * max_bounce_speed
                
                # Add some rotation from collision
                ball.angular_velocity += random.uniform(-10, 10)
        
        # Check ball-to-ball collisions
        for i in range(len(self.balls)):
            for j in range(i+1, len(self.balls)):
                if self.check_ball_collision(self.balls[i], self.balls[j]):
                    self.resolve_ball_collision(self.balls[i], self.balls[j])
        
        # Draw everything
        self.canvas.delete("all")
        self.heptagon.draw(self.canvas)
        for ball in self.balls:
            ball.draw(self.canvas)
        
        self.root.after(1000//FPS, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bouncing Balls in Spinning Heptagon")
    sim = Simulation(root)
    root.mainloop()