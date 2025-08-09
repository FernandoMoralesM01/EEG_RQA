import pygame
import math
import sys

# Setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎮 Xbox Controller Orb")
clock = pygame.time.Clock()

# Initialize joystick
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No controller connected.")
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using controller: {joystick.get_name()}")

# Player (Orb)
class Orb:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.radius = 20
        self.speed = 5
        self.glow = 100

    def update(self, dx, dy, pulse=False):
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))
        if pulse:
            self.glow = 255
        else:
            self.glow = max(100, self.glow - 5)

    def draw(self, surface):
        glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (0, 200, 255, self.glow), (int(self.x), int(self.y)), 40)
        surface.blit(glow_surface, (0, 0))
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), self.radius)

orb = Orb()

# Main loop
running = True
while running:
    screen.fill((10, 10, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read left stick axes
    pygame.event.pump()  # handle input
    axis_x = joystick.get_axis(0)  # left stick X
    axis_y = joystick.get_axis(1)  # left stick Y
    pulse = joystick.get_button(0)  # A button

    # Deadzone filtering
    if abs(axis_x) < 0.1: axis_x = 0
    if abs(axis_y) < 0.1: axis_y = 0

    orb.update(axis_x, axis_y, pulse)
    orb.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
