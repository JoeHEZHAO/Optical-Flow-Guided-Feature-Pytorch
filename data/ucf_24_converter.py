import os, sys

class_name = ['BasketballDunk', 'CliffDriving', 'CricketBowing', 'Fencing', 'IceDancing', 'LongJump', \
'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'Surfing', 'Basketball', \
'Biking', 'Diving', 'GolfSwing', 'HorseRiding', 'SoccerJuggling', 'TennisSwing', 'TrampolineJumping', \
'VolleyballSpiking', 'WalkingWithDog']

new_class_name = ['_' + x + '_' for x in class_name]

ucf_101_rgb_val_split1 = 'ucf101_rgb_val_split_1.txt'

new_list = []
with open(ucf_101_rgb_val_split1, 'r') as f:
    name_list = f.readlines()
    for l in name_list:
    	if any(sub_name in l for sub_name in new_class_name):
            # print(l)
            new_list.append(l)

for l in new_list:
    with open('ucf_24_rgb_val_split1.txt', 'a') as f:
        f.write(l)
