import os
import numpy as np


class CatchBall:
    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 16#8 shibasz
        self.screen_n_cols = 16#8 shibasz
        self.player_length = 6#3 shibasz
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5

        # variables
        self.reset()

    def next_ball(self):
        if np.random.rand() < 0.2:
            if self.ball_dir == 0:
                self.ball_dir = 1
            else:
                self.ball_dir = 0

        if self.ball_dir == 0:
            self.ball_row -= 1
            if self.ball_row < 0:
                self.ball_row = 1
                self.ball_dir = 1
        else:
            self.ball_row += 1
            if self.ball_row >= self.screen_n_rows:
                self.ball_row = self.screen_n_rows - 2
                self.ball_dir = 0
        return self.ball_row

    def update(self, action):
        """
        action:
            0: do nothing
            1: open buy #move left #up
            2: close buy #move right #down 
        """
        # update player position
        self.reward = 0
        self.terminal = False

        if action == self.enable_actions[1]:
            # open buy #move left #up
            if self.player_row < 0: #buy closed
                self.player_row = self.ball_row
                self.open_cnt = 0

        elif action == self.enable_actions[2]:
            # close buy #move right #down 
            if self.player_row >= 0: #buy opened


                # reward                
                #if self.player_row > self.ball_row:
                    # catch
                    #self.reward = 1
                #else:
                    # drop
                    #self.reward = -1
                self.reward = (self.player_row - self.ball_row)/self.screen_n_rows
                self.player_row = -1

                # reward #collision detection
                self.terminal = True
                
        else:
            # do nothing
            pass

        # time limit
        if self.player_row >= 0: #buy opened
            self.open_cnt += 1
            if self.open_cnt > self.screen_n_cols/2:
                self.reward = (self.player_row - self.ball_row)/self.screen_n_rows
                self.player_row = -1

                # reward #collision detection
                self.terminal = True

        # loss limit
        if self.player_row >= 0: #buy opened
            if (self.player_row - self.ball_row)/self.screen_n_rows < -1*self.screen_n_rows/3:
                self.reward = (self.player_row - self.ball_row)/self.screen_n_rows
                self.player_row = -1

                # reward #collision detection
                self.terminal = True



        # update ball position
        #self.ball_col += 1
        self.ball_hst = np.roll(self.ball_hst, -1)
        #if np.random.rand() < 0.5:
        #    self.ball_row = max(0, self.ball_row - 1)
        #else:
        #    self.ball_row = min(self.screen_n_rows-1, self.ball_row + 1)

        self.ball_row = self.next_ball()
        self.ball_hst[self.screen_n_cols-2] = self.ball_row

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw player
        #self.screen[self.player_row:self.player_row + self.player_length, self.player_col] = 1
        if self.player_row >= 0:
            self.screen[self.ball_row:self.player_row, self.player_col] = 1

        # draw ball
        #self.screen[self.ball_row, self.ball_col] = 1
        for i in range(self.screen_n_cols-1):
            self.screen[self.ball_hst[i], i] = 1

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_col = self.screen_n_cols - 1
        self.player_row = -1#np.random.randint(self.screen_n_rows - self.player_length)

        # reset ball position
        self.ball_col = self.screen_n_cols-2
        self.ball_row = np.random.randint(self.screen_n_rows) #np.int(self.screen_n_rows/2)
        self.ball_dir = 0

        # shiba price history
        self.ball_hst = np.empty(self.screen_n_cols-1, np.int32)

        for i in range(self.screen_n_cols-1):

            self.ball_hst = np.roll(self.ball_hst, -1)

            self.ball_row = self.next_ball()
            self.ball_hst[self.screen_n_cols-2] = self.ball_row


        # reset other variables
        self.reward = 0
        self.terminal = False


        self.open_cnt = 0
