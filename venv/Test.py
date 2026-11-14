import pygame
import random
import sys

pygame.init()

# Screen settings
WIDTH = 800
HEIGHT = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Timed Coin Collector")

clock = pygame.time.Clock()
FPS = 60

# Colors
SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (40, 100, 255)
YELLOW = (255, 215, 0)
DARK_YELLOW = (220, 160, 0)
GREEN = (40, 180, 80)
GRAY = (70, 70, 70)

# Fonts
font = pygame.font.SysFont(None, 32)
large_font = pygame.font.SysFont(None, 64)

# Game settings
PLAYER_SIZE = 35
PLAYER_X = 150
GAME_DURATION = 30  # Game length in seconds

# Movement states
UP = "up"
DOWN = "down"
STILL = "still"


class Player:
    def __init__(self):
        self.x = PLAYER_X
        self.y = HEIGHT // 2
        self.state = STILL
        self.speed = 5

    def update(self, keys):
        """
        The player is not constantly falling.

        Space held       -> move upward
        Down Arrow held  -> move downward
        Nothing held     -> remain still
        """

        if keys[pygame.K_SPACE]:
            self.state = UP
            self.y -= self.speed

        elif keys[pygame.K_DOWN]:
            self.state = DOWN
            self.y += self.speed

        else:
            self.state = STILL

        # Keep the player inside the screen.
        # Reaching the edge is not a game over.
        self.y = max(0, min(self.y, HEIGHT - PLAYER_SIZE))

    def get_rect(self):
        return pygame.Rect(
            self.x,
            int(self.y),
            PLAYER_SIZE,
            PLAYER_SIZE
        )

    def draw(self, surface):
        rect = self.get_rect()

        pygame.draw.rect(surface, BLUE, rect)

        # Eye
        pygame.draw.circle(
            surface,
            WHITE,
            (self.x + 25, int(self.y + 10)),
            5
        )


class Coin:
    def __init__(self):
        self.size = 18
        self.x = WIDTH + self.size
        self.y = random.randint(40, HEIGHT - 40)
        self.speed = random.randint(3, 6)

    def update(self):
        self.x -= self.speed

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.size),
            int(self.y - self.size),
            self.size * 2,
            self.size * 2
        )

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            DARK_YELLOW,
            (int(self.x), int(self.y)),
            self.size
        )

        pygame.draw.circle(
            surface,
            YELLOW,
            (int(self.x), int(self.y)),
            self.size - 4
        )


def draw_text(surface, text, font_object, color, x, y):
    text_surface = font_object.render(text, True, color)
    surface.blit(text_surface, (x, y))


def reset_game():
    player = Player()
    coins = []
    score = 0
    spawn_timer = 0
    start_time = pygame.time.get_ticks()
    game_over = False

    return player, coins, score, spawn_timer, start_time, game_over


player, coins, score, spawn_timer, start_time, game_over = reset_game()

running = True

while running:
    clock.tick(FPS)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_r and game_over:
                (
                    player,
                    coins,
                    score,
                    spawn_timer,
                    start_time,
                    game_over
                ) = reset_game()

    if not game_over:
        # Calculate remaining time
        elapsed_seconds = (pygame.time.get_ticks() - start_time) / 1000
        time_left = max(0, GAME_DURATION - elapsed_seconds)

        # End the game when time runs out
        if time_left <= 0:
            game_over = True

        # Update the player
        keys = pygame.key.get_pressed()
        player.update(keys)

        # Spawn coins periodically
        spawn_timer -= 1

        if spawn_timer <= 0:
            coins.append(Coin())

            # Coins appear sporadically
            spawn_timer = random.randint(30, 100)

        # Move coins
        for coin in coins:
            coin.update()

        # Remove coins that have moved off-screen
        coins = [
            coin for coin in coins
            if coin.x > -coin.size
        ]

        # Check for coin collection
        player_rect = player.get_rect()
        remaining_coins = []

        for coin in coins:
            if player_rect.colliderect(coin.get_rect()):
                score += 1
            else:
                remaining_coins.append(coin)

        coins = remaining_coins

    else:
        # Keep displaying the full timer after the game ends
        time_left = max(
            0,
            GAME_DURATION - (pygame.time.get_ticks() - start_time) / 1000
        )

    # Draw background
    screen.fill(SKY_BLUE)

    # Draw ground decoration
    pygame.draw.line(
        screen,
        GREEN,
        (0, HEIGHT - 5),
        (WIDTH, HEIGHT - 5),
        5
    )

    # Draw coins
    for coin in coins:
        coin.draw(screen)

    # Draw player
    player.draw(screen)

    # Draw score
    draw_text(
        screen,
        f"Coins: {score}",
        font,
        BLACK,
        20,
        20
    )

    # Draw player state
    draw_text(
        screen,
        f"State: {player.state.upper()}",
        font,
        BLACK,
        20,
        55
    )

    # Draw timer
    draw_text(
        screen,
        f"Time: {time_left:.1f}",
        font,
        BLACK,
        WIDTH - 150,
        20
    )

    # Instructions
    draw_text(
        screen,
        "Hold SPACE to move up",
        font,
        GRAY,
        20,
        HEIGHT - 70
    )

    draw_text(
        screen,
        "Hold DOWN to move down",
        font,
        GRAY,
        20,
        HEIGHT - 40
    )

    # Game-over screen
    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        game_over_text = large_font.render("TIME'S UP!", True, WHITE)
        score_text = font.render(
            f"You collected {score} coins",
            True,
            WHITE
        )
        restart_text = font.render(
            "Press R to play again or ESC to quit",
            True,
            WHITE
        )

        screen.blit(
            game_over_text,
            (
                WIDTH // 2 - game_over_text.get_width() // 2,
                HEIGHT // 2 - 90
            )
        )

        screen.blit(
            score_text,
            (
                WIDTH // 2 - score_text.get_width() // 2,
                HEIGHT // 2 - 20
            )
        )

        screen.blit(
            restart_text,
            (
                WIDTH // 2 - restart_text.get_width() // 2,
                HEIGHT // 2 + 35
            )
        )

    pygame.display.flip()

pygame.quit()
sys.exit()
