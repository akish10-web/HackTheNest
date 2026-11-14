import sys
import time
import random
import numpy as np
import pygame

from scipy.signal import butter, sosfiltfilt, welch


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 800
HEIGHT = 500
FPS = 60

GAME_DURATION = 30          # Game duration in seconds
PLAYER_SIZE = 35
PLAYER_X = 150
PLAYER_SPEED = 5

# Set this to True after your g.tec software is publishing EEG through LSL.
# Leave it False to test the game with the keyboard.
USE_LSL = False

# These frequencies must match the flickering targets in your SSVEP setup.
# Example:
#   10 Hz flicker = move up
#   15 Hz flicker = move down
UP_FREQUENCY = 10.0
DOWN_FREQUENCY = 15.0

# EEG settings
EEG_WINDOW_SECONDS = 1.0
MINIMUM_CONFIDENCE = 0.15
POWER_RATIO_THRESHOLD = 1.5

# If None, all EEG channels are used.
# You can change this to something such as [0, 1, 2, 3].
EEG_CHANNEL_INDICES = None


# ============================================================
# COLORS
# ============================================================

SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (40, 100, 255)
YELLOW = (255, 215, 0)
DARK_YELLOW = (220, 160, 0)
GREEN = (40, 180, 80)
GRAY = (70, 70, 70)
RED = (200, 40, 40)


# ============================================================
# ACTION STATES
# ============================================================

UP = "UP"
DOWN = "DOWN"
STILL = "STILL"


# ============================================================
# SSVEP DECODER
# ============================================================

class SSVEPDecoder:
    """
    Simple frequency-power SSVEP decoder.

    Input EEG must have this shape:

        channels x samples
    """

    def __init__(
        self,
        sample_rate,
        target_frequencies,
        channel_indices=None,
        window_seconds=1.0,
        minimum_confidence=0.15,
        power_ratio_threshold=1.5
    ):
        self.sample_rate = float(sample_rate)
        self.target_frequencies = list(target_frequencies)
        self.channel_indices = channel_indices
        self.window_samples = int(
            self.sample_rate * window_seconds
        )

        self.minimum_confidence = minimum_confidence
        self.power_ratio_threshold = power_ratio_threshold

        # Typical SSVEP frequency range.
        # Adjust if your experiment uses other frequencies.
        self.filter_sos = butter(
            4,
            [6.0, 40.0],
            btype="bandpass",
            fs=self.sample_rate,
            output="sos"
        )

    def predict(self, eeg):
        """
        Returns:

            action
            confidence
            frequency_scores

        eeg shape:

            channels x samples
        """

        if eeg is None or eeg.ndim != 2:
            return STILL, 0.0, {}

        if eeg.shape[1] < self.window_samples:
            return STILL, 0.0, {}

        # Use only the most recent EEG window.
        eeg = eeg[:, -self.window_samples:]

        # Select channels if requested.
        if self.channel_indices is not None:
            eeg = eeg[self.channel_indices, :]

        # Remove DC offset from every channel.
        eeg = eeg - np.mean(
            eeg,
            axis=1,
            keepdims=True
        )

        # Filter EEG.
        try:
            eeg = sosfiltfilt(
                self.filter_sos,
                eeg,
                axis=1
            )
        except ValueError:
            return STILL, 0.0, {}

        # Estimate power spectrum.
        frequencies, psd = welch(
            eeg,
            fs=self.sample_rate,
            nperseg=min(256, eeg.shape[1]),
            axis=1
        )

        frequency_scores = {}

        for target_frequency in self.target_frequencies:
            bandwidth = 0.75

            mask = (
                (frequencies >= target_frequency - bandwidth) &
                (frequencies <= target_frequency + bandwidth)
            )

            if not np.any(mask):
                power = 0.0
            else:
                # Average target-frequency power across channels.
                power = np.mean(psd[:, mask])

            frequency_scores[target_frequency] = float(power)

        if not frequency_scores:
            return STILL, 0.0, {}

        ranked = sorted(
            frequency_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        strongest_frequency, strongest_power = ranked[0]

        if len(ranked) > 1:
            second_power = ranked[1][1]
        else:
            second_power = 0.0

        if strongest_power <= 0:
            return STILL, 0.0, frequency_scores

        confidence = (
            strongest_power - second_power
        ) / (strongest_power + 1e-12)

        power_ratio = strongest_power / (
            second_power + 1e-12
        )

        # Reject uncertain classifications.
        if confidence < self.minimum_confidence:
            return STILL, confidence, frequency_scores

        if power_ratio < self.power_ratio_threshold:
            return STILL, confidence, frequency_scores

        if strongest_frequency == UP_FREQUENCY:
            return UP, confidence, frequency_scores

        if strongest_frequency == DOWN_FREQUENCY:
            return DOWN, confidence, frequency_scores

        return STILL, confidence, frequency_scores


class StableAction:
    """
    Prevents one noisy EEG prediction from immediately changing
    the player's movement.
    """

    def __init__(self, required_repetitions=2):
        self.required_repetitions = required_repetitions
        self.candidate_action = STILL
        self.stable_action = STILL
        self.repetitions = 0

    def update(self, action):
        if action == self.candidate_action:
            self.repetitions += 1
        else:
            self.candidate_action = action
            self.repetitions = 1

        if self.repetitions >= self.required_repetitions:
            self.stable_action = self.candidate_action

        return self.stable_action


# ============================================================
# LSL EEG CONNECTION
# ============================================================

def connect_to_lsl():
    """
    Connects to the first LSL stream whose type is EEG.

    Your g.tec software must already be configured to publish
    the EEG stream through Lab Streaming Layer.
    """

    try:
        from pylsl import resolve_byprop, StreamInlet
    except ImportError:
        raise RuntimeError(
            "pylsl is not installed. Run: pip install pylsl"
        )

    print("Searching for an EEG LSL stream...")

    streams = resolve_byprop(
        "type",
        "EEG",
        timeout=10
    )

    if not streams:
        raise RuntimeError(
            "No EEG LSL stream was found. "
            "Check that your g.tec software is publishing EEG through LSL."
        )

    inlet = StreamInlet(streams[0])
    info = inlet.info()

    sample_rate = info.nominal_srate()
    channel_count = info.channel_count()

    if sample_rate <= 0:
        raise RuntimeError(
            "The LSL stream does not provide a valid sample rate."
        )

    print("Connected to EEG LSL stream.")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {channel_count}")

    return inlet, sample_rate, channel_count


# ============================================================
# GAME OBJECTS
# ============================================================

class Player:
    def __init__(self):
        self.x = PLAYER_X
        self.y = HEIGHT // 2
        self.state = STILL

    def update(self, action):
        if action == UP:
            self.state = UP
            self.y -= PLAYER_SPEED

        elif action == DOWN:
            self.state = DOWN
            self.y += PLAYER_SPEED

        else:
            self.state = STILL

        # Touching the top or bottom is harmless.
        self.y = max(
            0,
            min(self.y, HEIGHT - PLAYER_SIZE)
        )

    def get_rect(self):
        return pygame.Rect(
            self.x,
            int(self.y),
            PLAYER_SIZE,
            PLAYER_SIZE
        )

    def draw(self, surface):
        rect = self.get_rect()

        pygame.draw.rect(
            surface,
            BLUE,
            rect
        )

        # Eye
        pygame.draw.circle(
            surface,
            WHITE,
            (self.x + 25, int(self.y + 10)),
            5
        )


class Coin:
    def __init__(self):
        self.radius = 18
        self.x = WIDTH + self.radius
        self.y = random.randint(
            40,
            HEIGHT - 40
        )
        self.speed = random.randint(3, 6)

    def update(self):
        self.x -= self.speed

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2
        )

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            DARK_YELLOW,
            (int(self.x), int(self.y)),
            self.radius
        )

        pygame.draw.circle(
            surface,
            YELLOW,
            (int(self.x), int(self.y)),
            self.radius - 4
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_text(surface, text, font, color, x, y):
    rendered = font.render(
        str(text),
        True,
        color
    )
    surface.blit(
        rendered,
        (x, y)
    )


def reset_game():
    return {
        "player": Player(),
        "coins": [],
        "score": 0,
        "spawn_timer": 0,
        "start_time": time.time(),
        "game_over": False
    }


# ============================================================
# PYGAME SETUP
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "SSVEP Coin Collector"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    None,
    28
)

small_font = pygame.font.SysFont(
    None,
    22
)

large_font = pygame.font.SysFont(
    None,
    64
)


# ============================================================
# EEG SETUP
# ============================================================

inlet = None
eeg_buffer = None
decoder = None
stabilizer = StableAction(
    required_repetitions=2
)

if USE_LSL:
    inlet, sample_rate, channel_count = connect_to_lsl()

    decoder = SSVEPDecoder(
        sample_rate=sample_rate,
        target_frequencies=[
            UP_FREQUENCY,
            DOWN_FREQUENCY
        ],
        channel_indices=EEG_CHANNEL_INDICES,
        window_seconds=EEG_WINDOW_SECONDS,
        minimum_confidence=MINIMUM_CONFIDENCE,
        power_ratio_threshold=POWER_RATIO_THRESHOLD
    )

    eeg_buffer = np.empty(
        (channel_count, 0),
        dtype=np.float64
    )


# ============================================================
# MAIN GAME LOOP
# ============================================================

game = reset_game()
running = True

while running:
    clock.tick(FPS)

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_r and game["game_over"]:
                game = reset_game()
                stabilizer = StableAction(
                    required_repetitions=2
                )

    player = game["player"]

    # --------------------------------------------------------
    # Determine player action
    # --------------------------------------------------------

    action = STILL
    confidence = 0.0
    frequency_scores = {}

    if not game["game_over"]:

        if USE_LSL:
            # Pull available EEG samples from the LSL stream.
            samples, timestamps = inlet.pull_chunk(
                timeout=0.0,
                max_samples=128
            )

            if samples:
                new_samples = np.asarray(
                    samples,
                    dtype=np.float64
                ).T

                eeg_buffer = np.concatenate(
                    [
                        eeg_buffer,
                        new_samples
                    ],
                    axis=1
                )

                # Keep only the latest two seconds.
                max_buffer_samples = int(
                    sample_rate * 2
                )

                if eeg_buffer.shape[1] > max_buffer_samples:
                    eeg_buffer = eeg_buffer[
                        :,
                        -max_buffer_samples:
                    ]

            # Decode the latest EEG window.
            action, confidence, frequency_scores = decoder.predict(
                eeg_buffer
            )

            # Smooth the classification.
            action = stabilizer.update(action)

        else:
            # Keyboard test mode.
            keys = pygame.key.get_pressed()

            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                action = UP

            elif keys[pygame.K_DOWN]:
                action = DOWN

            else:
                action = STILL

            confidence = 1.0

    # --------------------------------------------------------
    # Update timed game
    # --------------------------------------------------------

    if not game["game_over"]:
        elapsed_time = time.time() - game["start_time"]
        time_left = max(
            0.0,
            GAME_DURATION - elapsed_time
        )

        if time_left <= 0:
            game["game_over"] = True

        player.update(action)

        # Sporadically spawn coins.
        game["spawn_timer"] -= 1

        if game["spawn_timer"] <= 0:
            game["coins"].append(Coin())

            game["spawn_timer"] = random.randint(
                30,
                100
            )

        # Move coins.
        for coin in game["coins"]:
            coin.update()

        # Remove coins that left the screen.
        game["coins"] = [
            coin
            for coin in game["coins"]
            if coin.x > -coin.radius
        ]

        # Collect coins.
        player_rect = player.get_rect()
        remaining_coins = []

        for coin in game["coins"]:
            if player_rect.colliderect(coin.get_rect()):
                game["score"] += 1
            else:
                remaining_coins.append(coin)

        game["coins"] = remaining_coins

    else:
        elapsed_time = time.time() - game["start_time"]
        time_left = max(
            0.0,
            GAME_DURATION - elapsed_time
        )

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    screen.fill(SKY_BLUE)

    # Ground decoration
    pygame.draw.line(
        screen,
        GREEN,
        (0, HEIGHT - 5),
        (WIDTH, HEIGHT - 5),
        5
    )

    # Coins
    for coin in game["coins"]:
        coin.draw(screen)

    # Player
    player.draw(screen)

    # Information panel
    draw_text(
        screen,
        f"Coins: {game['score']}",
        font,
        BLACK,
        20,
        20
    )

    draw_text(
        screen,
        f"Time: {time_left:.1f}",
        font,
        BLACK,
        WIDTH - 140,
        20
    )

    draw_text(
        screen,
        f"Action: {player.state}",
        small_font,
        BLACK,
        20,
        58
    )

    if USE_LSL:
        draw_text(
            screen,
            f"Confidence: {confidence:.2f}",
            small_font,
            BLACK,
            20,
            85
        )

        if frequency_scores:
            score_text = " | ".join(
                [
                    f"{frequency:.0f}Hz: {power:.2e}"
                    for frequency, power
                    in frequency_scores.items()
                ]
            )

            draw_text(
                screen,
                score_text,
                small_font,
                BLACK,
                20,
                112
            )
    else:
        draw_text(
            screen,
            "Keyboard mode",
            small_font,
            BLACK,
            20,
            85
        )

    if not game["game_over"]:
        if USE_LSL:
            draw_text(
                screen,
                "SSVEP: use the configured visual targets",
                small_font,
                GRAY,
                20,
                HEIGHT - 45
            )
        else:
            draw_text(
                screen,
                "Hold SPACE/UP to move up; DOWN to move down",
                small_font,
                GRAY,
                20,
                HEIGHT - 45
            )

    # Game-over screen
    if game["game_over"]:
        overlay = pygame.Surface(
            (WIDTH, HEIGHT)
        )
        overlay.set_alpha(160)
        overlay.fill(BLACK)
        screen.blit(
            overlay,
            (0, 0)
        )

        title = large_font.render(
            "TIME'S UP!",
            True,
            WHITE
        )

        score_text = font.render(
            f"You collected {game['score']} coins",
            True,
            WHITE
        )

        restart_text = font.render(
            "Press R to restart or ESC to quit",
            True,
            WHITE
        )

        screen.blit(
            title,
            (
                WIDTH // 2 - title.get_width() // 2,
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
