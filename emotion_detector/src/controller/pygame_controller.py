import pygame
import pygame_widgets
from pygame_texteditor import TextEditor
from time import time, sleep
from pygame_widgets.button import Button

from view.main_view import MainView

WINDOW_SIZE = (1300, 800)
FPS = 30

# Text Area Constants
STYLE_FILE = "emotion_detector/asset/style/editor.yml"
FONT_SIZE = 21
OFFSET_X = 50
OFFSET_Y = 300
TEXTAREA_WIDTH = 600
TEXTAREA_HEIGHT = 300

# Buttons
COLOR_BUTTON = (190, 190, 190)
HOVER_BUTTON = (150, 150, 150)


class PygameController:
    def __init__(self) -> None:
        self.window_size = WINDOW_SIZE
        self.window = self.init_pygame()
        self.textarea = self.init_textarea()
        self.main_view = MainView(self.window, self.textarea, self.window_size)
        self.submit_button = self.init_submit_button()

    def pygame_loop(self) -> None:
        self.window = self.init_pygame()
        self.init_textarea()

        start = time()
        while True:
            events = pygame.event.get()

            if not self.is_running(events):
                break

            self.main_view.draw(events)
            pygame_widgets.update(events)
            pygame.display.flip()

            sleep(max(1.0/FPS - (time() - start), 0))
            continue

        self.quit_pygame()

    def is_running(self, events) -> bool:
        """
        Check if does not exist at least one of the following events: press ESQ or close the window

        Return:
            False if the events exist, True otherwise 
        """
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True

    def init_pygame(self) -> None:
        """
        Pygame and screen initialization 

        Return:
            Pygame Surface to display
        """
        pygame.init()
        return pygame.display.set_mode(self.window_size)

    def quit_pygame(self) -> None:
        '''Unitialize all pygame modules'''
        pygame.quit()

    def init_textarea(self):
        textarea = TextEditor(
            OFFSET_X, OFFSET_Y, TEXTAREA_WIDTH, TEXTAREA_HEIGHT, self.window)
        textarea.set_font_size(FONT_SIZE)
        textarea.set_syntax_highlighting(True)
        textarea.set_colorscheme_from_yaml(STYLE_FILE)
        return textarea

    def init_submit_button(self):
        return Button(
            self.window, 100, 650, 150, 50,
            text='SUBMIT',
            fontSize=35,
            margin=20,
            inactiveColour=COLOR_BUTTON,
            hoverColour=HOVER_BUTTON,
            pressedColour=HOVER_BUTTON,
            radius=20,
            onClick=lambda: print('Click')
        )
