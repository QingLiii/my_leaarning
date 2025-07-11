import tkinter as tk
import math
import numpy as np
import random
from dataclasses import dataclass
from typing import List
import sys
import time

# Constants ------------------------------------------------------------
WIDTH, HEIGHT = 800, 800
CENTER = np.array([WIDTH / 2, HEIGHT / 2])
FPS = 60
GRAVITY = 0.3
FRICTION = 0.99
BALL_RADIUS = 15
HEPTAGON_RADIUS = 300
SPIN_PERIOD = 5  # seconds per full revolution
BALL_COUNT = 20
COLORS = [
    "#f8b862", "#f6ad49", "#f39800", "#f08300", "#ec6d51", "#ee7948", "#ed6d3d",
    "#ec6800", "#ec6800", "#ee7800", "#eb6238", "#ea5506", "#ea5506", "#eb6101",
    "#e49e61", "#e45e32", "#e17b34", "#dd7a56", "#db8449", "#d66a35"
]

# Ball -----------------------------------------------------------------
@dataclass
class Ball:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    color: str
    ang: float = 0.0  # angle of number rotation
    ang_vel: float = 0.0

    def step(self, dt):
        self.vel[1] += GRAVITY * dt
        self.vel *= FRICTION
        self.pos += self.vel * dt
        self.ang += self.ang_vel * dt
        self.ang_vel *= FRICTION

# Geometry helpers -----------------------------------------------------
def heptagon_vertices(center, radius, angle_offset):
    vertices = []
    for i in range(7):
        theta = angle_offset + i * 2 * math.pi / 7
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        vertices.append(np.array([x, y]))
    return vertices

def point_to_line_distance(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj), proj

def circle_line_collision(ball: Ball, a: np.ndarray, b: np.ndarray):
    dist, point = point_to_line_distance(ball.pos, a, b)
    if dist < ball.radius:
        normal = (ball.pos - point) / dist
        overlap = ball.radius - dist
        ball.pos += normal * overlap
        dot = np.dot(ball.vel, normal)
        ball.vel -= 2 * dot * normal
        # Limit bounce energy
        max_speed = math.sqrt(2 * GRAVITY * (HEPTAGON_RADIUS - ball.radius))
        speed = np.linalg.norm(ball.vel)
        if speed > max_speed:
            ball.vel = ball.vel / speed * max_speed
        # Add spin
        ball.ang_vel += random.uniform(-5, 5)

def circle_circle_collision(b1: Ball, b2: Ball):
    delta = b2.pos - b1.pos
    dist = np.linalg.norm(delta)
    if dist < b1.radius + b2.radius:
        normal = delta / dist
        overlap = b1.radius + b2.radius - dist
        b1.pos -= normal * overlap / 2
        b2.pos += normal * overlap / 2
        # Elastic collision
        m1, m2 = 1, 1
        v1, v2 = b1.vel, b2.vel
        v1_new = v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, normal) * normal
        v2_new = v2 - (2 * m1 / (m1 + m2)) * np.dot(v2 - v1, normal) * normal
        b1.vel, b2.vel = v1_new, v2_new
        # Spin transfer
        spin_total = b1.ang_vel + b2.ang_vel
        b1.ang_vel = spin_total / 2 + random.uniform(-3, 3)
        b2.ang_vel = spin_total / 2 + random.uniform(-3, 3)

# Main -----------------------------------------------------------------
class Simulation:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.balls = []
        self.angle = 0
        self.dt = 1 / FPS
        self.init_balls()
        self.update()

    def init_balls(self):
        for i in range(BALL_COUNT):
            pos = np.array([CENTER[0] + random.uniform(-1, 1),
                            CENTER[1] + random.uniform(-1, 1)])
            vel = np.array([random.uniform(-2, 2), random.uniform(-2, 0)])
            ball = Ball(id=i+1,
                        pos=pos,
                        vel=vel,
                        radius=BALL_RADIUS,
                        color=COLORS[i])
            self.balls.append(ball)

    def update(self):
        self.angle += 2 * math.pi / (SPIN_PERIOD * FPS)
        # Move balls
        for ball in self.balls:
            ball.step(self.dt)
        # Ball-wall collisions
        vertices = heptagon_vertices(CENTER, HEPTAGON_RADIUS, self.angle)
        for ball in self.balls:
            for i in range(7):
                a = vertices[i]
                b = vertices[(i + 1) % 7]
                circle_line_collision(ball, a, b)
        # Ball-ball collisions
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                circle_circle_collision(self.balls[i], self.balls[j])
        # Draw
        self.canvas.delete("all")
        # Draw heptagon
        v = [tuple(v) for v in vertices]
        self.canvas.create_polygon(*v, outline="white", width=3, fill="")
        # Draw balls
        for ball in self.balls:
            x, y = ball.pos
            self.canvas.create_oval(x - ball.radius, y - ball.radius,
                                    x + ball.radius, y + ball.radius,
                                    fill=ball.color, outline="white", width=2)
            # Draw number in the ball, rotated
            self.canvas.create_text(x, y,
                                    text=str(ball.id),
                                    fill="white",
                                    font=("Arial", 12, "bold"),
                                    angle=math.degrees(ball.ang))
        self.root.after(int(1000 / FPS), self.update)

# Run ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spinning Heptagon Bounce")
    Simulation(root)
    root.mainloop()