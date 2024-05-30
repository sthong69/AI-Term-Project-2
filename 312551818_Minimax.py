import pygame
import math
import sys
import os 
import random

pygame.init()

# Set environment variable to ensure the window opens at the center of the screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Initialize Pygame
pygame.init()

# Get current screen resolution
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Create a window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define colors and hexagon properties
BG_COLOR = (30, 30, 30)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
HEX_SIZE = 30
HEX_BORDER = 2
PIECE_RADIUS = int(HEX_SIZE * 0.8)

# Initialize font for text rendering
pygame.font.init()
font = pygame.font.SysFont(None, int(HEX_SIZE * 0.7))

# Initialize game state variables
hexagon_board = {}
selected_counts = {}
turn_ended = False
max_selected_counts = {}
initial_counts = {}  # Store initial counts for each label

def draw_player_turn_button(screen, button_text, message=""):
    """Draws a turn indicator button at the top right of the screen."""
    screen_width, screen_height = screen.get_size()
    font = pygame.font.SysFont(None, 36)  # Set the font size to 36
    button_width, button_height = 150, 50  # Width and height of the button
    button_x = screen_width - button_width - 10  # 10 pixels margin from the right edge
    button_y = 10  # 10 pixels margin from the top edge
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Draw a blue button

    # Check if the button text is 'Game Over'
    if button_text == 'Game Over':
        text = button_text
    else:
        text = button_text + "'s turn"
    
    text_surf = font.render(text, True, pygame.Color('white'))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))

    return button_rect

def draw_hexagon(surface, x, y, size, border_color, fill_color, number=None, border_thickness=2, number_color=BLACK):
    """Draws a hexagon with optional number in its center."""
    angles_deg = [60 * i + 30 for i in range(6)]
    outer_points = [(x + (size + border_thickness) * math.cos(math.radians(angle)),
                     y + (size + border_thickness) * math.sin(math.radians(angle))) for angle in angles_deg]
    inner_points = [(x + size * math.cos(math.radians(angle)),
                     y + size * math.sin(math.radians(angle))) for angle in angles_deg]
    
    if fill_color == WHITE:
        border_color = GRAY  
        number_color = WHITE  
    elif fill_color == BLACK:
        border_color = WHITE  
        
    pygame.draw.polygon(surface, border_color, outer_points)
    pygame.draw.polygon(surface, fill_color, inner_points)

    if number is not None:
        text_surface = font.render(str(number), True, number_color)
        text_rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, text_rect)


def point_in_hex(x, y, hex_x, hex_y, size):
    """Check if the point (x, y) is inside the hexagon centered at (hex_x, hex_y)."""
    dx = abs(x - hex_x)
    dy = abs(y - hex_y)
    return dx <= size * math.sqrt(3) / 2 and dy <= size * 3 / 2 and size * 3 / 2 - dx * math.sqrt(3) / 3 > dy

def draw_hex_shape_grid(surface, center_row, center_col, size):
    """Draws a grid of hexagons on the screen, labeled by distance from center."""
    global hexagon_board
    initial_counts.clear()  

    def get_hex_label(row, col, max_dist):
        dist_from_center = max(abs(row), abs(col), abs(row + col))
        if dist_from_center == 0:
            return 1
        label = 5
        
        corners = [(row, col) for row in (-max_dist, max_dist) for col in (-max_dist, max_dist)]
        corners.extend([(-max_dist, 0), (0, -max_dist), (max_dist, 0), (0, max_dist)])

        if (row, col) in corners:
            label = 6
        elif abs(row) == max_dist or abs(col) == max_dist or abs(row + col) == max_dist:
            label = 5
        elif abs(row) == max_dist - 1 or abs(col) == max_dist - 1 or abs(row + col) == max_dist - 1:
            label = 3
        elif abs(row) <= max_dist - 2 and abs(col) <= max_dist - 2 and abs(row + col) <= max_dist - 2:
            label = 2
       
        return label

    max_dist = center_row

    for row in range(-center_row, center_row + 1):
        for col in range(-center_col, center_col + 1):
            dist_from_center = max(abs(row), abs(col), abs(row + col))
            if dist_from_center <= center_row:
                x = WIDTH / 2 + (col + row / 2) * (math.sqrt(3) * (size + HEX_BORDER))
                y = HEIGHT / 2 + row * ((size + HEX_BORDER) * 1.5)
                label = get_hex_label(row, col, max_dist)
                
                # hexagon_board[(row, col)] = {'x': x, 'y': y, 'label': label, 'selected': False} 
                hexagon_board[(row, col)] = {
                                                'x': x,
                                                'y': y,
                                                'label': label,
                                                'selected': False,
                                                'owner': None  # Track which player has selected the hexagon
                                            }
                
                initial_counts[label] = initial_counts.get(label, 0) + 1

                draw_hexagon(surface, x, y, size, (255, 255, 255), (255, 228, 205), label)

def draw_piece(surface, center_x, center_y, color):
    """Draws a game piece at specified coordinates."""
    pygame.draw.circle(surface, color, (int(center_x), int(center_y)), PIECE_RADIUS)

def draw_end_turn_button(screen):
    # Draw a simple button on the screen and return its rect
    font = pygame.font.SysFont(None, 36)
    button_rect = pygame.Rect(650, 550, 150, 50)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Blue button
    text_surf = font.render('End Turn', True, pygame.Color('white'))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))
    return button_rect

def check_all_hexes_selected():
    """Checks if all hexes on the board have been selected."""
    return all(hex_info['selected'] for hex_info in hexagon_board.values())

def calculate_connected_areas(color):
    """Calculates the largest connected area of hexes of the specified color."""
    def dfs(row, col, visited):
        if (row, col) in visited or not (row, col) in hexagon_board or hexagon_board[(row, col)]['owner'] != color:
            return 0
        visited.add((row, col))
        count = 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            next_row, next_col = row + dr, col + dc
            count += dfs(next_row, next_col, visited)
        return count

    visited = set()
    max_area = 0
    for (row, col), info in hexagon_board.items():
        if (row, col) not in visited and info['owner'] == color:
            area = dfs(row, col, visited)
            if area > max_area:
                max_area = area
    return max_area

def display_connected_areas():
    """Displays the size of the largest connected areas for black and white pieces."""
    black_area = calculate_connected_areas(1)  # 黑子的標籤為1
    white_area = calculate_connected_areas(2)  # 白子的標籤為2
    black_text = font.render(f"Black Area: {black_area}", True, BLACK)
    white_text = font.render(f"White Area: {white_area}", True, WHITE)
    screen.blit(black_text, (10, 10))
    screen.blit(white_text, (10, 30))
    pygame.display.update()

def update_selected_hexes(selected_hexes):
    """Updates the state and visual representation of selected hexes."""
    global current_turn, selected_counts
    for hex_info in selected_hexes:
        hex_info['selected'] = True
        hex_info['booked'] = True
        hex_info['owner'] = current_turn  # Update the owner when a hex is selected
        selected_counts[hex_info['label']] = selected_counts.get(hex_info['label'], 0) + 1
        fill_color = BLACK if current_turn == 'black' else WHITE
        draw_hexagon(screen, hex_info['x'], hex_info['y'], HEX_SIZE, (128, 128, 128), fill_color, hex_info['label'])
        pygame.display.flip()

def process_selections_by_round(x, y, current_round):
    """Processes player selections on the board based on the current round number."""
    global current_label, required_selections
    selected_hexes = []
    
    for (hx, hy), hex_info in hexagon_board.items():
        if point_in_hex(x, y, hex_info['x'], hex_info['y'], HEX_SIZE) and not hex_info.get('booked', False):
            if current_round == 1 and hex_info['label'] == 2 and selected_counts.get(2, 0) < 1:
                selected_hexes.append(hex_info)
                break  # In the first round, only one hexagon labeled as 2 is allowed to be selected.
            elif current_round > 1:
                if current_label is None:
                    current_label = hex_info['label']
                    required_selections = hex_info['label']
                if current_label == hex_info['label'] and selected_counts.get(current_label, 0) < required_selections:
                    selected_hexes.append(hex_info)
                if selected_counts.get(current_label, 0) >= required_selections:
                    break  # Stop selecting once the required number is reached.

    return selected_hexes

def display_remaining_hexes():
    """Calculates and displays the remaining number of hexes for each label on the terminal."""
    remaining_counts = {1: 0, 2: 0, 3: 0, 5: 0, 6: 0} # Initialize counters
    
    # Traverse the hexagon_board to update the remaining count for each label
    for hex_info in hexagon_board.values():
        if not hex_info['selected']:  # If the hex is not selected
            label = hex_info['label']
            if label in remaining_counts:
                remaining_counts[label] += 1
    
    # Output the remaining count for each label
    all_selected = True
    #print("Remaining hexes by label:")
    for label, count in sorted(remaining_counts.items()):
        #print(f"Label {label}: {count}")
        if count > 0:
            all_selected = False
            # break
    return all_selected
            
def auto_select_remaining_hexes(label, required_selections):
    """Automatically selects the remaining hexes for a label if the turn timer expires."""
    hexes_by_label = {}
    # Collect all unbooked hexes of the current label
    for pos, info in hexagon_board.items():
        if not info.get('booked', False) and info['label'] == label:
            hexes_by_label.setdefault(label, []).append(info)
    
    remaining_hexes = hexes_by_label.get(label, [])
    number_to_select = min(len(remaining_hexes), required_selections - selected_counts.get(label, 0))

    selected_hexes = random.sample(remaining_hexes, number_to_select) if number_to_select > 0 else []
    # Process selected hexes
    update_selected_hexes(selected_hexes)

    print(f"AI selected {number_to_select} hexes automatically due to timeout.")     

def Count_Hexagons_by_Owner():
    owner_count = {'None': 0, 'white': 0, 'black': 0}
    for hex_info in hexagon_board.values():
        owner = hex_info['owner']
        if owner is None:
            owner_count['None'] += 1
        else:
            owner_count[owner] += 1
    
    print(owner_count)


def end_current_round():
    """Ends the current round, checking if the conditions for round completion are met."""
    global current_label, current_turn, turn_ended, current_round, selected_counts, hexagon_board
       
    if current_label is None:
        if current_round != 1:
            print("No selections have been made this round. You must select at least one hex.")
            return  # Do not end the round if no selections have been made
    # # Before ending the round, calculate the remaining and selected counts
    remaining_hexes = len([info for info in hexagon_board.values() if info['label'] == current_label and not info.get('booked', False)])
    selected_hexes_count = selected_counts.get(current_label, 0)

    if (remaining_hexes == 0) and (selected_hexes_count + 1) == current_label:
        
        print(f"Ending round with sufficient selections for label: {current_label}.")
        display_remaining_hexes()
        reset_round_state()    
        return
    
    # Check if the round can be ended
    elif current_label is not None:
        
        if selected_hexes_count != current_label:
            
            print(f"You have not selected enough hexes of label: {current_label}. Needed: {current_label}, selected: {selected_hexes_count}.")
            return 
         
        elif remaining_hexes < current_label:
            if selected_hexes_count < remaining_hexes:
                print(f"Not enough hexes left for label: {current_label}. You need to select all {remaining_hexes} available hexes.")
                # Additional steps might be needed to cancel the originally booked hexes
                return  # Do not end the round if not all available hexes are selected when fewer than needed
        # All conditions are met to end the round
        elif selected_hexes_count < current_label:
            # Additional steps might be needed to cancel the originally booked hexes
            print(f"You have not selected enough hexes of label: {current_label}. Needed: {current_label}, selected: {selected_hexes_count}.")
    
    # All conditions for round completion are satisfied
    display_remaining_hexes()
    print(f"Ending round with sufficient selections for label: {current_label}.")
    reset_round_state()


def reset_round_state():
    """Resets the state of the round, preparing for the next one."""
    global current_label, current_turn, turn_ended, current_round, selected_counts
    current_label = None
    selected_counts = {}
    print("Round ended, it's now " + current_turn + "'s turn.")
    current_turn = 'white' if current_turn == 'black' else 'black'
    turn_ended = True
    draw_player_turn_button(screen, current_turn)
    current_round += 1
    pygame.display.flip()

def select_hexes_by_random(hexes_by_label, current_round):
    import copy 
    """FIXED: Calculates the largest connected area of hexes of the specified color."""
    def calculate_connected_areas_v2(board_state, color):
        def dfs(row, col, visited):
            if (row, col) in visited or not (row, col) in board_state or board_state[(row, col)]['owner'] != color:
                return 0
            visited.add((row, col))
            count = 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                next_row, next_col = row + dr, col + dc
                count += dfs(next_row, next_col, visited)
            return count

        visited = set()
        max_area = 0
        for (row, col), info in board_state.items():
            if (row, col) not in visited and info['owner'] == color:
                area = dfs(row, col, visited)
                if area > max_area:
                    max_area = area
        return max_area
    
    def combinations(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)
    
    def get_moves_by_label(board_state):
        moves_by_label = {}
        for pos, info in board_state.items():
            if not info.get('booked', False):
                label = info['label']
                if label in moves_by_label:
                    moves_by_label[label].append((pos, info))
                else:
                    moves_by_label[label] = [(pos, info)]
        return moves_by_label
    
    def get_possible_moves(board_state):
        moves_by_label = get_moves_by_label(board_state)
        possible_moves = []
        for label in moves_by_label:
            available_hexes = [couple[0] for couple in moves_by_label[label]]
            for limit in range(1,int(label)+1):
                possibilities = list(combinations(available_hexes, limit))
                possibilities_labeled = [[label] + list(possibility) for possibility in possibilities]
                possible_moves += possibilities_labeled
        sorted_moves = sorted(possible_moves, key=len)
        return sorted_moves[::-1]
            
# 5: [((-5, 1), {'x': 316.8615612366939, 'y': 60.0, 'label': 5, 'selected': False, 'owner': None}), 
# ((-5, 2), {'x': 372.28718707889794, 'y': 60.0, 'label': 5, 'selected': False, 'owner': None}), 
# ((-5, 3), {'x': 427.71281292110206, 'y': 60.0, 'label': 5, 'selected': False, 'owner': None}),
        
    def get_new_state(board_state, moves, color):
        res = copy.deepcopy(board_state)
        for hex in moves:
            res[hex]['selected'] = True
            res[hex]['booked'] = True
            res[hex]['owner'] = color
        return res
        
    def get_score(board_state, color):
        # If it is obviously a win or a lose (WIP PAS OUF)
        remaining_moves_by_label = get_moves_by_label(board_state)
        maximum_number_of_points = 0
        for label in remaining_moves_by_label:
            for (hex,hex_info) in remaining_moves_by_label[label]:
                maximum_number_of_points += 1
        maximum_number_of_points = round(maximum_number_of_points/2)
        
        black_connected_areas = calculate_connected_areas_v2(board_state, 'black')
        white_connected_areas = calculate_connected_areas_v2(board_state, 'white')
        
        if color == 'black':
            if black_connected_areas >= white_connected_areas:
                if black_connected_areas-white_connected_areas > maximum_number_of_points:
                    return 10000000000
            if black_connected_areas < white_connected_areas:
                if white_connected_areas-black_connected_areas > maximum_number_of_points:
                    return -10000000000
        if color == 'white':
            if black_connected_areas >= white_connected_areas:
                if black_connected_areas-white_connected_areas > maximum_number_of_points:
                    return -10000000000
            if black_connected_areas < white_connected_areas:
                if white_connected_areas-black_connected_areas > maximum_number_of_points:
                    return 10000000000
                
        # Evaluation function if it is not obviously a win or a lose
        def get_all_possesed_nodes(board_state, color):
            res = []
            for hex in board_state:
                if board_state[hex]['owner'] == color:
                    res.append(hex)
            return res
        
        def compute_avg_distance(hexes):
            distance = 0
            for hex1 in hexes:
                for hex2 in hexes:
                    distance += abs(hex1[0]-hex2[0]) + abs(hex1[1]-hex2[0])+1
            return 10000000000/distance        
        
        def score_function(color):
            if color == 'white':
                return white_connected_areas**5 - black_connected_areas**5 + compute_avg_distance(get_all_possesed_nodes(board_state,color))
            if color == 'black':
                return black_connected_areas**5 - white_connected_areas**5 + compute_avg_distance(get_all_possesed_nodes(board_state,color))
        
        score = score_function(color)
        #print(color + str(score))
        return score
        
    def is_terminal(board_state):
        board_size = len(board_state)
        res = 0
        for hex in board_state:
            if board_state[hex]['selected']:
                res +=1
        return res == board_size
    
    def get_opponent(color):
        if color == 'black':
            return 'white'
        if color == 'white':
            return 'black'
                
    def minimax(board_state, depth, should_be_maxed, color, alpha, beta):
        board_state_copy = copy.deepcopy(board_state)
        opponent = get_opponent(color)
        label = 0
            
        if (depth==0) or (is_terminal(board_state_copy)):
            return get_score(board_state_copy, color), None

        if should_be_maxed:
            value = float('-inf')
            possible_moves = get_possible_moves(board_state_copy)
            possible_moves = possible_moves[:10]
            for i in range(len(possible_moves)):
                tmp_label = possible_moves[i][0]
                moves = possible_moves[i][1:]
                child = get_new_state(board_state_copy, moves, color)

                tmp = minimax(child, depth-1, False, opponent, alpha, beta)[0]
                if tmp > value:
                    value = tmp
                    best_movement = moves
                    label = tmp_label
                # Alpha-Beta Prunning
                alpha = max(alpha,value)
                if (beta <= alpha):
                    print("Pruned!")
                    break
        else:
            value = float('inf')
            possible_moves = get_possible_moves(board_state_copy)
            possible_moves = possible_moves[:10]
            for i in range(len(possible_moves)):
                tmp_label = possible_moves[i][0]
                moves = possible_moves[i][1:]
                child = get_new_state(board_state_copy, moves, opponent)

                tmp = minimax(child, depth-1, True, color, alpha, beta)[0]
                if tmp < value:
                    value = tmp
                    best_movement = moves
                    label = tmp_label
                # Alpha-Beta Prunning
                beta = min(beta,value)
                if (beta <= alpha):
                    print("Pruned!")
                    break
        return value, best_movement, label

    selected_hexes = []
    if current_round <= 1:
        # Special handling for the first round: only select one hexagon labeled as 2.
        if 2 in hexes_by_label and any(not hex_info['selected'] for _, hex_info in hexes_by_label[2]):
            available_hexes = [(pos, hex_info) for pos, hex_info in hexes_by_label[2] if not hex_info['selected']]
            if available_hexes:
                selected_hexes.append(random.choice(available_hexes))
    else:
        board_state = copy.deepcopy(hexagon_board)
        value, chosen_hexes, label = minimax(board_state, 3, True, current_turn, float('-inf'), float('inf'))
        print("Best final game state evaluation: "+ str(value))
        print(chosen_hexes)
        print(label)
        
        for final_couple in hexes_by_label[label]:
            pos, hex_info = final_couple
            if pos in chosen_hexes:
                selected_hexes.append((pos,hex_info))
    return selected_hexes

# {(-1, -1): {'x': 316.8615612366939, 'y': 252.0, 'label': 2, 'selected': False, 'owner': None}
# (3, -3): {'x': 316.8615612366939, 'y': 444.0, 'label': 2, 'selected': True, 'owner': 'white', 'booked': True}
# (-1, 1): {'x': 427.71281292110206, 'y': 252.0, 'label': 2, 'selected': True, 'owner': 'black', 'booked': True}}

# hexes by label
# {1: [((0, 0), {'x': 400.0, 'y': 300.0, 'label': 1, 'selected': False, 'owner': None} ) ] }

def check_timeout_and_autocomplete():
    """Checks for turn timeout and auto-completes selection if necessary."""
    global start_time
    current_time = pygame.time.get_ticks()
    if ((current_turn == 'black' and black_player_type == "human") or
        (current_turn == 'white' and white_player_type == "human")):
        if (current_time - start_time) > 30000:  # If the count exceeds 30, automatically complete the move.
            auto_select_remaining_hexes(current_label, required_selections)
            end_current_round()
            start_time = pygame.time.get_ticks()  # 重置起始時間    
 
def main(black_player, white_player):
    # Your existing implementation of the game logic here
    print(f"Player 1: {black_player}, Player 2: {white_player}")
    
    """Main game loop."""
    global selected_counts, current_label, current_turn, turn_ended, current_round, \
           start_time, required_selections, black_player_type, white_player_type, remaining_hexes, \
           hexagon_board, running  
    start_time = pygame.time.get_ticks()  # 記錄起始時間
    running = True
    current_round = 1
    current_label = None  # Track the label selected in the current round
    required_selections = 0  # Required selections based on the first selected label
    clock = pygame.time.Clock()
    black_player_type = black_player
    white_player_type = white_player
    current_turn = 'black'

    # Set the game window background color and update the display
    screen.fill(BG_COLOR)
    draw_hex_shape_grid(screen, 5, 5, HEX_SIZE)
    pygame.display.flip()

    # Draw the 'End Turn' button and update the display
    button_rect = draw_end_turn_button(screen)
    draw_player_turn_button(screen, current_turn)
    pygame.display.flip()
    
    # Define a custom event.
    KEEP_ALIVE_EVENT = pygame.USEREVENT + 1
    # Set up a timer to trigger a custom event every 1 second
    pygame.time.set_timer(KEEP_ALIVE_EVENT, 1000)
    
    
    # Start the main game loop
    while running:
        check_timeout_and_autocomplete()
        # Handle all events in the game
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                # Exit the game if the window is closed
                pygame.quit()
                sys.exit()
            elif event.type == KEEP_ALIVE_EVENT:
                break
                #print("Keep alive event triggered.")
                
            # Handle mouse click events for players set as "Human"
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                    
                if (current_turn == 'black' and black_player_type == "human") or \
                   (current_turn == 'white' and white_player_type == "human"):
                   
                    # 手動選擇座標
                    selected_hexes = process_selections_by_round(x, y, current_round)
                    # 更新座標
                    update_selected_hexes(selected_hexes)
                    
                # Handle the 'End Turn' button click
                if button_rect.collidepoint(event.pos):
                    # Count_Hexagons_by_Owner()
                    end_current_round()
                        
            # Handle random selections for players set as "Random"
            if not turn_ended and ((current_turn == 'black' and black_player_type == "random") or \
                                   (current_turn == 'white' and white_player_type == "random")):
                # Filter unbooked hexes from the hexagon_board and categorize them by label
                hexes_by_label = {}
                for pos, info in hexagon_board.items():
                    if not info.get('booked', False):
                        label = info['label']
                        if label in hexes_by_label:
                            hexes_by_label[label].append((pos, info))
                        else:
                            hexes_by_label[label] = [(pos, info)]
                    
                # Randomly select a specified number of hexes from those filtered by label
                selected_hexes = select_hexes_by_random(hexes_by_label, current_round)
                
                # Process the selected hexes
                for pos, hex_info in selected_hexes:
                    hex_info['selected'] = True
                    hex_info['booked'] = True
                    hex_info['owner'] = current_turn
                    fill_color = BLACK if current_turn == 'black' else WHITE
                    draw_hexagon(screen, hex_info['x'], hex_info['y'], HEX_SIZE, (128, 128, 128), fill_color, hex_info['label'])
                
                print("black connected: "+str(calculate_connected_areas('black')))
                print("white connected: "+str(calculate_connected_areas('white')))
                pygame.display.flip()
                pygame.time.wait(100)  # Wait a second to let players see AI's choice
                #Count_Hexagons_by_Owner()    
                # time.sleep(1)
                current_turn = 'white' if current_turn == 'black' else 'black'
                turn_ended = True
                draw_player_turn_button(screen, current_turn)
                pygame.display.flip()
            
                display_remaining_hexes()
                # Update the round number
                current_round += 1
            
            turn_ended = False
        
        if display_remaining_hexes():
            print("Game Over: All hexes have been selected.")
            draw_player_turn_button(screen, "Game Over")
            running = False  # Exit the main loop
            pygame.display.flip()
            clock.tick(30)
            
        # Update the display and control the game update rate
        pygame.display.flip()
        clock.tick(30)
          
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py [player1_type] [player2_type]")
        print("player1_type and player2_type should be 'human' or 'random'")
        sys.exit(1)  # Exit the script with an error code

    main(sys.argv[1], sys.argv[2])
    # main('random', 'random')