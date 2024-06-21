# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

# "large" interval means [-1, 1], "short" means [0, 1]
interval_large = [False, False, True, True, False, False, False, False, False, False,
     True, True, True, True, True, True, False, False, False, True,
     False, False, False, False, False, False, False, True, True, True,
     True, False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, True, False, False, False, False, 
     True, True, False, False, False, False]

interval_short = [not x for x in interval_large]

def rig_control_remap(control_tensor):
    """
    Replicates the animators interface to prevent conflicting controls
    
    control_tensor.shape should be: (batch_size, 56)
    
    [-1, 1] controls are indicated by "True"
    [0, 1] controls are indicated by "False"
    
    ["browDown_L", "browDown_R", ("cheekSuck_L", "cheekPuff_L"), ("cheekSuck_R", "cheekPuff_R"), "cheekRaiser_L", "cheekRaiser_R", "chinRaiser_B", "chinRaiser_T", "dimpler_L", "dimpler_R",
     ("eyesClosed_L, upperLidRaiser_L), ("eyesClosed_R, upperLidRaiser_R), (eyesLookDown_L, eyesLookUp_L), (eyesLookDown_R, eyesLookUp_R), (eyesLookLeft_L, eyesLookRight_L), (eyesLookLeft_R, eyesLookRight_R), innerBrowRaiser_L, innerBrowRaiser_R, jawDrop, (jawSidewaysLeft, jawSidewaysRight),
     "jawThrust", "lidTightener_L", "lidTightener_R", "lipCornerDepressor_L", "lipCornerDepressor_R", "lipCornerPuller_L", "lipCornerPuller_R", ("lipSuck_LB", "lipFunneler_LB"), ("lipSuck_LT", "lipFunneler_LT"), ("lipSuck_RB", "lipFunneler_RB"),
     ("lipSuck_RT", "lipFunneler_RT"), "lipPressor_L", "lipPressor_R", "lipPucker_L", "lipPucker_R", "lipsToward_LB", "lipsToward_LT", "lipsToward_RB", "lipsToward_RT", "lipStretcher_L", 
     "lipStretcher_R", "lipTightener_L", "lipTightener_R", "mouthLowerDown_L", "mouthLowerDown_R, ("mouthRight", "mouthLeft"), "nasolabialFurrow_L", "nasolabialFurrow_R", "noseWrinkler_L", "noseWrinkler_R", 
     ("nostrilCompressor_L", "nostrilDilator_L"), ("nostrilCompressor_R", "nostrilDilator_R"), "outerBrowRaiser_L", "outerBrowRaiser_R", "upperLipRaiser_L", "upperLipRaiser_R"]
    
    [False, False, True, True, False, False, False, False, False, False,
     True, True, True, True, True, True, False, False, False, True,
     False, False, False, False, False, False, False, True, True, True,
     True, False, False, False, False, False, False, False, False, False,
     False, False, False, False, False, True, False, False, False, False, 
     True, True, False, False, False, False]
     
    out_coeffs.shape: (batch_size, 72)
    """
    
    #control_tensor = torch.tensor(in_controls).float().to(device)

    browDown_L = torch.clamp(control_tensor[:, 0], min=0, max=1)
    browDown_R = torch.clamp(control_tensor[:, 1], min=0, max=1)
    cheekPuff_L = torch.clamp(control_tensor[:, 2], min=0, max=1) #linked with cheekSuck_L
    cheekPuff_R = torch.clamp(control_tensor[:, 3], min=0, max=1) #linked with cheekSuck_R
    cheekRaiser_L = torch.clamp(control_tensor[:, 4], min=0, max=1)
    cheekRaiser_R = torch.clamp(control_tensor[:, 5], min=0, max=1)
    cheekSuck_L = torch.clamp(control_tensor[:, 2], min=-1, max=0)*-1 #linked with cheekPuff_L
    cheekSuck_R = torch.clamp(control_tensor[:, 3], min=-1, max=0)*-1 #linked with cheekPuff_R
    chinRaiser_B = torch.clamp(control_tensor[:, 6], min=0, max=1)
    chinRaiser_T = torch.clamp(control_tensor[:, 7], min=0, max=1)
    dimpler_L = torch.clamp(control_tensor[:, 8], min=0, max=1)
    dimpler_R = torch.clamp(control_tensor[:, 9], min=0, max=1)
    eyesClosed_L = torch.clamp(control_tensor[:, 10], min=0, max=1) #linked with upperLidRaiser_L
    eyesClosed_R = torch.clamp(control_tensor[:, 11], min=0, max=1) #linked with upperLidRaiser_R
    eyesLookDown_L = torch.clamp(control_tensor[:, 12], min=-1, max=0)*-1 #linked with eyesLookUp_L
    eyesLookDown_R = torch.clamp(control_tensor[:, 13], min=-1, max=0)*-1 #linked with eyesLookUp_R
    eyesLookLeft_L = torch.clamp(control_tensor[:, 14], min=-1, max=0)*-1 #linked with eyesLookRight_L
    eyesLookLeft_R = torch.clamp(control_tensor[:, 15], min=-1, max=0)*-1 #linked with eyesLookRight_R
    eyesLookRight_L = torch.clamp(control_tensor[:, 14], min=0, max=1) #linked with eyesLookLeft_L
    eyesLookRight_R = torch.clamp(control_tensor[:, 15], min=0, max=1) #linked with eyesLookLeft_R
    eyesLookUp_L = torch.clamp(control_tensor[:, 12], min=0, max=1) #linked with eyesLookDown_L
    eyesLookUp_R = torch.clamp(control_tensor[:, 13], min=0, max=1) #linked with eyesLookDown_R
    innerBrowRaiser_L = torch.clamp(control_tensor[:, 16], min=0, max=1)
    innerBrowRaiser_R = torch.clamp(control_tensor[:, 17], min=0, max=1)
    jawDrop = torch.clamp(control_tensor[:, 18], min=0, max=1)
    jawSidewaysLeft = torch.clamp(control_tensor[:, 19], min=-1, max=0)*-1 #linked with jawSidewaysRight
    jawSidewaysRight = torch.clamp(control_tensor[:, 19], min=0, max=1) #linked with jawSidewaysLeft
    jawThrust = torch.clamp(control_tensor[:, 20], min=0, max=1)
    lidTightener_L = torch.clamp(control_tensor[:, 21], min=0, max=1)
    lidTightener_R = torch.clamp(control_tensor[:, 22], min=0, max=1)
    lipCornerDepressor_L = torch.clamp(control_tensor[:, 23], min=0, max=1)
    lipCornerDepressor_R = torch.clamp(control_tensor[:, 24], min=0, max=1)
    lipCornerPuller_L = torch.clamp(control_tensor[:, 25], min=0, max=1)
    lipCornerPuller_R = torch.clamp(control_tensor[:, 26], min=0, max=1)
    lipFunneler_LB = torch.clamp(control_tensor[:, 27], min=0, max=1) #linked with lipSuck_LB
    lipFunneler_LT = torch.clamp(control_tensor[:, 28], min=0, max=1) #linked with lipSuck_LT
    lipFunneler_RB = torch.clamp(control_tensor[:, 29], min=0, max=1) #linked with lipSuck_RB
    lipFunneler_RT = torch.clamp(control_tensor[:, 30], min=0, max=1) #linked with lipSuck_RT
    lipPressor_L = torch.clamp(control_tensor[:, 31], min=0, max=1)
    lipPressor_R = torch.clamp(control_tensor[:, 32], min=0, max=1)
    lipPucker_L = torch.clamp(control_tensor[:, 33], min=0, max=1)
    lipPucker_R = torch.clamp(control_tensor[:, 34], min=0, max=1)
    lipsToward_LB = torch.clamp(control_tensor[:, 35], min=0, max=1)
    lipsToward_LT = torch.clamp(control_tensor[:, 36], min=0, max=1)
    lipsToward_RB = torch.clamp(control_tensor[:, 37], min=0, max=1)
    lipsToward_RT = torch.clamp(control_tensor[:, 38], min=0, max=1)
    lipStretcher_L = torch.clamp(control_tensor[:, 39], min=0, max=1)
    lipStretcher_R = torch.clamp(control_tensor[:, 40], min=0, max=1)
    lipSuck_LB = torch.clamp(control_tensor[:, 27], min=-1, max=0)*-1 #linked with lipFunneler_LB
    lipSuck_LT = torch.clamp(control_tensor[:, 28], min=-1, max=0)*-1 #linked with lipFunneler_LT
    lipSuck_RB = torch.clamp(control_tensor[:, 29], min=-1, max=0)*-1 #linked with lipFunneler_RB
    lipSuck_RT = torch.clamp(control_tensor[:, 30], min=-1, max=0)*-1 #linked with lipFunneler_RT
    lipTightener_L = torch.clamp(control_tensor[:, 41], min=0, max=1)
    lipTightener_R = torch.clamp(control_tensor[:, 42], min=0, max=1)
    mouthLowerDown_L = torch.clamp(control_tensor[:, 43], min=0, max=1)
    mouthLowerDown_R = torch.clamp(control_tensor[:, 44], min=0, max=1)
    mouthLeft = torch.clamp(control_tensor[:, 45], min=0, max=1) #linked with mouthRight
    mouthRight = torch.clamp(control_tensor[:, 45], min=-1, max=0)*-1 #linked with mouthLeft
    nasolabialFurrow_L = torch.clamp(control_tensor[:, 46], min=0, max=1)
    nasolabialFurrow_R = torch.clamp(control_tensor[:, 47], min=0, max=1)
    noseWrinkler_L = torch.clamp(control_tensor[:, 48], min=0, max=1)
    noseWrinkler_R = torch.clamp(control_tensor[:, 49], min=0, max=1)
    nostrilCompressor_L = torch.clamp(control_tensor[:, 50], min=-1, max=0)*-1 #linked with nostrilDilator_L
    nostrilCompressor_R = torch.clamp(control_tensor[:, 51], min=-1, max=0)*-1 #linked with nostrilDilator_R
    nostrilDilator_L = torch.clamp(control_tensor[:, 50], min=0, max=1) #linked with nostrilCompressor_L
    nostrilDilator_R = torch.clamp(control_tensor[:, 51], min=0, max=1) #linked with nostrilCompressor_R
    outerBrowRaiser_L = torch.clamp(control_tensor[:, 52], min=0, max=1)
    outerBrowRaiser_R = torch.clamp(control_tensor[:, 53], min=0, max=1)
    upperLidRaiser_L = torch.clamp(control_tensor[:, 10], min=-1, max=0)*-1 #linked with eyesClosed_L
    upperLidRaiser_R = torch.clamp(control_tensor[:, 11], min=-1, max=0)*-1 #linked with eyesClosed_R
    upperLipRaiser_L = torch.clamp(control_tensor[:, 54], min=0, max=1)
    upperLipRaiser_R = torch.clamp(control_tensor[:, 55], min=0, max=1)
    
    out_coeff = torch.stack([
                browDown_L,
                browDown_R,
                cheekPuff_L,
                cheekPuff_R,
                cheekRaiser_L,
                cheekRaiser_R,
                cheekSuck_L,
                cheekSuck_R,
                chinRaiser_B,
                chinRaiser_T,
                dimpler_L,
                dimpler_R,
                eyesClosed_L,
                eyesClosed_R,
                eyesLookDown_L,
                eyesLookDown_R,
                eyesLookLeft_L,
                eyesLookLeft_R,
                eyesLookRight_L,
                eyesLookRight_R,
                eyesLookUp_L,
                eyesLookUp_R,
                innerBrowRaiser_L,
                innerBrowRaiser_R,
                jawDrop,
                jawSidewaysLeft,
                jawSidewaysRight,
                jawThrust,
                lidTightener_L,
                lidTightener_R,
                lipCornerDepressor_L,
                lipCornerDepressor_R,
                lipCornerPuller_L,
                lipCornerPuller_R,
                lipFunneler_LB,
                lipFunneler_LT,
                lipFunneler_RB,
                lipFunneler_RT,
                lipPressor_L,
                lipPressor_R,
                lipPucker_L,
                lipPucker_R,
                lipsToward_LB,
                lipsToward_LT,
                lipsToward_RB,
                lipsToward_RT,
                lipStretcher_L,
                lipStretcher_R,
                lipSuck_LB,
                lipSuck_LT,
                lipSuck_RB,
                lipSuck_RT,
                lipTightener_L,
                lipTightener_R,
                mouthLowerDown_L,
                mouthLowerDown_R,
                mouthLeft,
                mouthRight,
                nasolabialFurrow_L,
                nasolabialFurrow_R,
                noseWrinkler_L,
                noseWrinkler_R,
                nostrilCompressor_L,
                nostrilCompressor_R,
                nostrilDilator_L,
                nostrilDilator_R,
                outerBrowRaiser_L,
                outerBrowRaiser_R,
                upperLidRaiser_L,
                upperLidRaiser_R,
                upperLipRaiser_L,
                upperLipRaiser_R
                ], axis = 1)
    return out_coeff



def piecewise_linear(x, x1, x2, x3):
    return (x>=x1) * (x<x2) * ((x2 - x)/(x1-x2) + 1.0) + ((x>=x2) * (x<=x3) * ((x - x2)/(x2-x3) + 1.0))

def rig_logic(control_tensor):
    rig_logic_tensor = torch.empty(control_tensor.shape[0], 287).to(control_tensor.device)
    rig_logic_tensor[:,:72] = control_tensor
    
    rig_logic_tensor[:,72] = piecewise_linear(control_tensor[:,24], x1=0.5, x2=0.75, x3=1.0)
        
    rig_logic_tensor[:,[73, 74, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 102, 103]] = \
        piecewise_linear(control_tensor[:,[2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 54, 55]], 
                         x1=0.0, x2=0.5, x3=1.0)

    rtmp = piecewise_linear(control_tensor[:,[12, 13, 14, 15]], x1=0.0, x2=0.5, x3=1.0)
    
    rig_logic_tensor[:,[75, 76, 92, 93, 100, 101]] = \
        piecewise_linear(control_tensor[:,[4, 5, 32, 33, 46, 47]], x1=.25, x2=0.5, x3=1.0)
               
    rig_logic_tensor[:,87] = piecewise_linear(control_tensor[:,24], x1=0.25, x2=0.5, x3=0.75)
        
    rig_logic_tensor[:,range(104,111)] = piecewise_linear(control_tensor[:,[4, 5, 24, 32, 33, 46, 47]], x1=0.0, x2=0.25, x3=0.5)
    
    rig_logic_tensor[:,range(111, 115)] = piecewise_linear(control_tensor[:,[48, 49, 50, 51]], x1=0.0, x2=0.25, x3=1.0)

    rig_logic_tensor[:,range(115, 221)] = control_tensor[:,[4,5,12,13,14,15,22,23,2,3,12,13,14,15,28,29,32,33,60,61,66,67,10,11,10,11,10,11,10,11,12,13,12,13,12,13,12,13,12,13,12,13,12,13,16,17,18,19,22,23,28,29,66,67,68,69,16,17,20,21,24,24,32,33,34,35,36,37,40,41,46,47,54,55,70,71,32,33,28,29,32,33,70,71,46,47,54,55,32,33,70,71,40,40,41,41,54,54,55,55,54,55,70,71,70,71]] * \
        control_tensor[:,[0,1,0,1,0,1,0,1,24,24,4,5,4,5,4,5,4,5,4,5,4,5,24,24,32,33,46,47,54,55,14,15,16,17,18,19,20,21,22,23,28,29,66,67,14,15,14,15,14,15,14,15,14,15,14,15,20,21,18,19,30,31,24,24,24,24,24,24,24,24,24,24,24,24,24,24,28,29,68,69,30,31,30,31,32,33,32,33,58,59,32,33,34,35,36,37,34,35,36,37,46,47,58,59,60,61]]
    
    rig_logic_tensor[:,[221,222]] = rtmp[:,[0,1]] * control_tensor[:,[14,15]]
    rig_logic_tensor[:,range(223,229)] = control_tensor[:,[22,23,66,67,12,13]] * rtmp[:,[0,1,0,1,2,3]]
    rig_logic_tensor[:,range(229,233)] = rtmp[:,[2,3,2,3]] * control_tensor[:,[22,23,66,67]]
    
    rig_logic_tensor[:,range(233,277)] = control_tensor[:,[12,13,12,13,12,13,12,13,14,15,32,33,10,11,10,11,12,13,12,13,12,13,46,47,54,55,70,71,40,40,41,41,34,35,36,37,40,40,41,41,46,47,70,71]] * \
        control_tensor[:,[14,15,14,15,28,29,32,33,28,29,28,29,32,33,46,47,22,23,28,29,66,67,32,33,32,33,32,33,34,35,36,37,24,24,24,24,42,43,44,45,32,33,32,33]] * \
        control_tensor[:,[0,1,4,5,4,5,4,5,4,5,4,5,24,24,32,33,14,15,14,15,14,15,24,24,24,24,24,24,24,24,24,24,42,43,44,45,24,24,24,24,54,55,58,59]]
    
    rig_logic_tensor[:,[277,278]] = rtmp[:,[2,3]] * control_tensor[:,[22,23]] * rtmp[:,[0,1]]    
    rig_logic_tensor[:,[279,280]] = rtmp[:,[2,3]] * rtmp[:,[0,1]] * control_tensor[:,[66,67]]    
    rig_logic_tensor[:, range(281, 287)] = control_tensor[:,[12,13,40,40,41,41]] * control_tensor[:,[4,5,34,35,36,37]] * control_tensor[:,[28,29,24,24,24,24]] * control_tensor[:,[0,1,42,43,44,45]]
    
    return rig_logic_tensor

fixed_shapes_map_v2 = [[ 0,  1], [1., 1.],    # 1, browLowerer
 [ 2,  3], [1., 1.],    # 2, cheekPuff
 [ 6,  7], [1., 1.],    # 3, chinRaiser
 [ 8,  9], [1., 1.],    # 4, dimpler
 [10, 11], [1., 1.],    # 5, eyesClosed
 [12, 13], [-1., -1.],  # 6, eyesLookDown
 [14, 15], [-1., -1.],  # 7, eyesLookLeft
 [14, 15], [1., 1.],    # 8, eyesLookRight
 [12, 13], [1., 1.],    # 9, eyesLookUp
 [16, 17], [1., 1.],    # 10, innerBrowRaiser
 [18], [1.],            # 11, jawDrop
 [21, 22], [1., 1.],    # 12, lidTightener
 [23, 24], [1., 1.],	# 13, lipCornerDepressor
 [25, 26], [1., 1.],    # 14, lipCornerPuller
 [27, 28, 29, 30], [1., 1., 1., 1.], # 15, lipFunneler
 [33, 34], [1., 1.],	# 16, lipPucker
 [39, 40], [1., 1.],	# 17, lipStretcher
 [27, 28, 29, 30], [-1., -1., -1., -1.], # 18, lipSuck
 [41, 42], [1., 1.],	# 19, lipTightener
 [43, 44], [1., 1.],    # 20, lowerLipDepressor
 [45], [1.],            # 21, mouthLeft
 [45], [-1.],           # 22, mouthRight
 [46, 47], [1., 1.],    # 23, nasolabialFurrow
 [48, 49], [1., 1.],    # 24, noseWrinkler
 [52, 53], [1., 1.],	# 25, outerBrowRaiser
 [10, 11], [-1., -1.],  # 26, upperLidRaiser
 [54, 55], [1., 1.],    # 27, upperLipRaiser
]


fixed_shapes_map_v1 = [[ 0,  1], [1., 1.],    # 1, browLowerer
 [ 2,  3], [1., 1.],    # 2, cheekPuff
 [ 8,  9], [1., 1.],    # 3, dimpler
 [10, 11], [1., 1.],    # 4, eyesClosed
 [12, 13], [-1., -1.],  # 5, eyesLookDown
 [14, 15], [-1., -1.],  # 6, eyesLookLeft
 [14, 15], [1., 1.],    # 7, eyesLookRight
 [12, 13], [1., 1.],    # 8, eyesLookUp
 [16, 17], [1., 1.],    # 9, innerBrowRaiser
 [21, 22], [1., 1.],    # 10, lidTightener
 [23, 24], [1., 1.],	# 11, lipCornerDepressor
 [25, 26], [1., 1.],    # 12, lipCornerPuller
 [27, 28, 29, 30], [1., 1., 1., 1.], # 13, lipFunneler
 [33, 34], [1., 1.],	# 14, lipPucker
 [39, 40], [1., 1.],	# 15, lipStretcher
 [27, 28, 29, 30], [-1., -1., -1., -1.], # 16, lipSuck
 [41, 42], [1., 1.],	# 17, lipTightener
 [43, 44], [1., 1.],    # 18, lowerLipDepressor
 [45], [1.],            # 19, mouthLeft
 [45], [-1.],           # 20, mouthRight
 [46, 47], [1., 1.],    # 21, nasolabialFurrow
 [52, 53], [1., 1.],	# 22, outerBrowRaiser
 [10, 11], [-1., -1.],  # 23, upperLidRaiser
 [54, 55], [1., 1.],    # 24, upperLipRaiser
]


def compute_rig_logic(control_tensor, inbetween_dict, corrective_dict):
    control_size = control_tensor.shape[1]
    rig_logic_size = control_size + inbetween_dict['num_indices'] + corrective_dict['num_indices']

    rig_logic_tensor = torch.empty(control_tensor.shape[0], rig_logic_size).to(control_tensor.device)
    rig_logic_tensor[:, :control_size] = control_tensor

    # solve inbetweens first since correctives can depend on the inbetween values
    for inbetween_values in inbetween_dict.keys():
        if inbetween_values == "num_indices": continue

        driver_range_mask = inbetween_dict[inbetween_values][0]
        driven_range_mask = inbetween_dict[inbetween_values][1]

        inbetween_values_list = list(inbetween_values.split(','))
        inbetween_values_list = list(map(float, inbetween_values_list))

        # vectorized compute
        rig_logic_tensor[:, driver_range_mask] = piecewise_linear(control_tensor[:, driven_range_mask],
                                                                  x1=inbetween_values_list[0],
                                                                  x2=inbetween_values_list[1],
                                                                  x3=inbetween_values_list[2])

    tmp_tensor = rig_logic_tensor  # must copy to tmp in order to avoid a CUDA memory error when accessing and assigning to the same tensor

    # solve correctives last
    for num_mults in corrective_dict.keys():
        if num_mults == "num_indices": continue

        driver_range_mask = corrective_dict[num_mults][-1]
        corrective_value = tmp_tensor[:, corrective_dict[num_mults][0]]
        for j in range(1, num_mults):
            corrective_value = corrective_value * tmp_tensor[:, corrective_dict[num_mults][j]]
        rig_logic_tensor[:, driver_range_mask] = corrective_value

    return rig_logic_tensor