from typing import Tuple

import numpy as np
import pygame
from PIL import Image

from csgo.action_processing import CSGOAction
from .csgo_play_env import CsgoPlayEnv


class CsgoGame:
    def __init__(
        self,
        play_env: CsgoPlayEnv,
        size: Tuple[int, int],
        mouse_multiplier: int = 10,
        fps: int = 15,
        verbose: bool = True,
        fullscreen: bool = False,
    ) -> None:
        self.env = play_env
        self.height, self.width = size
        self.mouse_multiplier = mouse_multiplier
        self.fps = fps
        self.verbose = verbose
        self.fullscreen = fullscreen

        self.env.print_controls()
        print("\nGeneral:\n")
        print(" .  : pause/unpause")
        print(" e  : step when paused")
        print(" ⏎  : reset")
        print("Esc : quit\n")

    def run(self) -> None:
        pygame.init()
        header_height = 120 if self.verbose else 0
        font_size = 16
        if self.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            x_center, y_center = screen.get_rect().center
            header_width = 540
            x_header = x_center - header_width // 2
            y_header = y_center - self.height // 2 - header_height - 10
            header_rect = pygame.Rect(x_header, y_header, header_width, header_height)
        else:
            screen = pygame.display.set_mode((self.width, self.height + header_height))
            x_center = y_center = 0
            header_width = self.width
            header_rect = pygame.Rect(0, 0, self.width, header_height)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, idx_column, num_cols):
            x_pos = 5 + idx_column * int(header_width // num_cols)
            y_pos = 5 + idx_line * font_size
            screen.blit(
                font.render(text, True, pygame.Color("white")),
                (header_rect.x + x_pos, header_rect.y + y_pos),
            )

        def draw_obs(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            if self.fullscreen:
                screen.blit(surface, (x_center - self.width // 2, y_center - self.height // 2))
            else:
                screen.blit(surface, (0, header_height))

        def reset():
            nonlocal obs, info, do_reset, keys_pressed, l_click, r_click
            obs, info = self.env.reset()
            pygame.event.clear()
            do_reset = False
            keys_pressed = []
            l_click = r_click = False

        obs, info, do_reset = None, None, False
        keys_pressed, l_click, r_click = [], False, False
        reset()
        do_wait = False
        should_stop = False

        while not should_stop:
            do_one_step = False
            mouse_x, mouse_y = 0, 0
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    should_stop = True

                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.rel
                    mouse_x *= self.mouse_multiplier
                    mouse_y *= self.mouse_multiplier

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        l_click = True
                    if event.button == 3:
                        r_click = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        l_click = False
                    if event.button == 3:
                        r_click = False

                if event.type == pygame.KEYDOWN:
                    keys_pressed.append(event.key)
                elif event.type == pygame.KEYUP and event.key in keys_pressed:
                    keys_pressed.remove(event.key)

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_RETURN:
                    do_reset = True
                if event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                if event.key == pygame.K_e:
                    do_one_step = True
                if event.key == pygame.K_m:
                    do_reset = self.env.next_mode()
                if event.key == pygame.K_UP:
                    do_reset = self.env.next_axis_1()
                if event.key == pygame.K_DOWN:
                    do_reset = self.env.prev_axis_1()

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            csgo_action = CSGOAction(keys_pressed, mouse_x, mouse_y, l_click, r_click)
            next_obs, rew, end, trunc, info = self.env.step(csgo_action)

            draw_obs(obs)
            if self.verbose and info is not None:
                clear_header()
                header = info["header"]
                num_cols = len(header)
                for j, col in enumerate(header):
                    for i, row in enumerate(col):
                        draw_text(row, idx_line=i, idx_column=j, num_cols=num_cols)

            pygame.display.flip()
            clock.tick(self.fps)

            if end or trunc:
                reset()
            else:
                obs = next_obs

        pygame.quit()
