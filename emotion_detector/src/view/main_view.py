import pygame

BG_COLOR = (252, 252, 220)


class MainView:
    def __init__(self, window, textarea, window_size) -> None:
        self.window = window
        self.window_size = window_size
        self.textarea = textarea

    def draw(self, events):
        pressed_keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        self.window.fill(BG_COLOR)

        self.textarea.display_editor(
            events, pressed_keys, mouse_x, mouse_y, mouse_pressed)
