TRAIN_FILES = ['../data//Adiac_TRAIN',  # 0
               '../data//ArrowHead_TRAIN',  # 1
               '../data//ChlorineConcentration_TRAIN',  # 2
               '../data//InsectWingbeatSound_TRAIN',  # 3
               '../data//Lighting7_TRAIN',  # 4
               '../data//Wine_TRAIN',  # 5
               '../data//WordsSynonyms_TRAIN',  # 6
               '../data//50words_TRAIN',  # 7
               '../data//Beef_TRAIN',  # 8
               '../data//DistalPhalanxOutlineAgeGroup_TRAIN',  # 9
               '../data//DistalPhalanxOutlineCorrect_TRAIN',  # 10
               '../data//DistalPhalanxTW_TRAIN',  # 11
               '../data//ECG200_TRAIN',  # 12
               '../data//ECGFiveDays_TRAIN',  # 13
               '../data//BeetleFly_TRAIN',  # 14
               '../data//BirdChicken_TRAIN',  # 15
               '../data//ItalyPowerDemand_TRAIN',  # 16
               '../data//SonyAIBORobotSurface_TRAIN',  # 17
               '../data//SonyAIBORobotSurfaceII_TRAIN',  # 18
               '../data//MiddlePhalanxOutlineAgeGroup_TRAIN',  # 19
               '../data//MiddlePhalanxOutlineCorrect_TRAIN',  # 20
               '../data//MiddlePhalanxTW_TRAIN',  # 21
               '../data//ProximalPhalanxOutlineAgeGroup_TRAIN',  # 22
               '../data//ProximalPhalanxOutlineCorrect_TRAIN',  # 23
               '../data//ProximalPhalanxTW_TRAIN',  # 24
               '../data//MoteStrain_TRAIN',  # 25
               '../data//MedicalImages_TRAIN',  # 26
               '../data//Strawberry_TRAIN',  # 27
               '../data//ToeSegmentation1_TRAIN',  # 28
               '../data//Coffee_TRAIN',  # 29
               '../data//Cricket_X_TRAIN',  # 30
               '../data//Cricket_Y_TRAIN',  # 31
               '../data//Cricket_Z_TRAIN',  # 32
               '../data//uWaveGestureLibrary_X_TRAIN',  # 33
               '../data//uWaveGestureLibrary_Y_TRAIN',  # 34
               '../data//uWaveGestureLibrary_Z_TRAIN',  # 35
               '../data//ToeSegmentation2_TRAIN',  # 36
               '../data//DiatomSizeReduction_TRAIN',  # 37
               '../data//Car_TRAIN',  # 38
               '../data//CBF_TRAIN',  # 39
               '../data//CinC_ECG_torso_TRAIN',  # 40
               '../data//Computers_TRAIN',  # 41
               '../data//Earthquakes_TRAIN',  # 42
               '../data//ECG5000_TRAIN',  # 43
               '../data//ElectricDevices_TRAIN',  # 44
               '../data//FaceAll_TRAIN',  # 45
               '../data//FaceFour_TRAIN',  # 46
               '../data//FacesUCR_TRAIN',  # 47
               '../data//Fish_TRAIN',  # 48
               '../data//FordA_TRAIN',  # 49
               '../data//FordB_TRAIN',  # 50
               '../data//Gun_Point_TRAIN',  # 51
               '../data//Ham_TRAIN',  # 52
               '../data//HandOutlines_TRAIN',  # 53
               '../data//Haptics_TRAIN',  # 54
               '../data//Herring_TRAIN',  # 55
               '../data//InlineSkate_TRAIN',  # 56
               '../data//LargeKitchenAppliances_TRAIN',  # 57
               '../data//Lighting2_TRAIN',  # 58
               '../data//Mallat_TRAIN',  # 59
               '../data//Meat_TRAIN',  # 60
               '../data//NonInvasiveFatalECG_Thorax1_TRAIN',  # 61
               '../data//NonInvasiveFatalECG_Thorax2_TRAIN',  # 62
               '../data//OliveOil_TRAIN',  # 63
               '../data//OSULeaf_TRAIN',  # 64
               '../data//PhalangesOutlinesCorrect_TRAIN',  # 65
               '../data//Phoneme_TRAIN',  # 66
               '../data//Plane_TRAIN',  # 67
               '../data//RefrigerationDevices_TRAIN',  # 68
               '../data//ScreenType_TRAIN',  # 69
               '../data//ShapeletSim_TRAIN',  # 70
               '../data//ShapesAll_TRAIN',  # 71
               '../data//SmallKitchenAppliances_TRAIN',  # 72
               '../data//StarLightCurves_TRAIN',  # 73
               '../data//SwedishLeaf_TRAIN',  # 74
               '../data//Symbols_TRAIN',  # 75
               '../data//synthetic_control_TRAIN',  # 76
               '../data//Trace_TRAIN',  # 77
               '../data//Two_Patterns_TRAIN',  # 78
               '../data//TwoLeadECG_TRAIN',  # 79
               '../data//UWaveGestureLibraryAll_TRAIN',  # 80
               '../data//Wafer_TRAIN',  # 81
               '../data//Worms_TRAIN',  # 82
               '../data//WormsTwoClass_TRAIN',  # 83
               '../data//Yoga_TRAIN',  # 84
               '../data//ACSF1_TRAIN',  # 85
               '../data//AllGestureWiimoteX_TRAIN',  # 86
               '../data//AllGestureWiimoteY_TRAIN',  # 87
               '../data//AllGestureWiimoteZ_TRAIN',  # 88
               '../data//BME_TRAIN',  # 89
               '../data//Chinatown_TRAIN',  # 90
               '../data//Crop_TRAIN',  # 91
               '../data//DodgerLoopDay_TRAIN',  # 92
               '../data//DodgerLoopGame_TRAIN',  # 93
               '../data//DodgerLoopWeekend_TRAIN',  # 94
               '../data//EOGHorizontalSignal_TRAIN',  # 95
               '../data//EOGVerticalSignal_TRAIN',  # 96
               '../data//EthanolLevel_TRAIN',  # 97
               '../data//FreezerRegularTrain_TRAIN',  # 98
               '../data//FreezerSmallTrain_TRAIN',  # 99
               '../data//Fungi_TRAIN',  # 100
               '../data//GestureMidAirD1_TRAIN',  # 101
               '../data//GestureMidAirD2_TRAIN',  # 102
               '../data//GestureMidAirD3_TRAIN',  # 103
               '../data//GesturePebbleZ1_TRAIN',  # 104
               '../data//GesturePebbleZ2_TRAIN',  # 105
               '../data//GunPointAgeSpan_TRAIN',  # 106
               '../data//GunPointMaleVersusFemale_TRAIN',  # 107
               '../data//GunPointOldVersusYoung_TRAIN',  # 108
               '../data//HouseTwenty_TRAIN',  # 109
               '../data//InsectEPGRegularTrain_TRAIN',  # 110
               '../data//InsectEPGSmallTrain_TRAIN',  # 111
               '../data//MelbournePedestrian_TRAIN',  # 112
               '../data//MixedShapesRegularTrain_TRAIN',  # 113
               '../data//MixedShapesSmallTrain_TRAIN',  # 114
               '../data//PickupGestureWiimoteZ_TRAIN',  # 115
               '../data//PigAirwayPressure_TRAIN',  # 116
               '../data//PigArtPressure_TRAIN',  # 117
               '../data//PigCVP_TRAIN',  # 118
               '../data//PLAID_TRAIN',  # 119
               '../data//PowerCons_TRAIN',  # 120
               '../data//Rock_TRAIN',  # 121
               '../data//SemgHandGenderCh2_TRAIN',  # 122
               '../data//SemgHandMovementCh2_TRAIN',  # 123
               '../data//SemgHandSubjectCh2_TRAIN',  # 124
               '../data//ShakeGestureWiimoteZ_TRAIN',  # 125
               '../data//SmoothSubspace_TRAIN',  # 126
               '../data//UMD_TRAIN'  # 127
               ]

TEST_FILES = ['../data//Adiac_TEST',  # 0
              '../data//ArrowHead_TEST',  # 1
              '../data//ChlorineConcentration_TEST',  # 2
              '../data//InsectWingbeatSound_TEST',  # 3
              '../data//Lighting7_TEST',  # 4
              '../data//Wine_TEST',  # 5
              '../data//WordsSynonyms_TEST',  # 6
              '../data//50words_TEST',  # 7
              '../data//Beef_TEST',  # 8
              '../data//DistalPhalanxOutlineAgeGroup_TEST',  # 9
              '../data//DistalPhalanxOutlineCorrect_TEST',  # 10
              '../data//DistalPhalanxTW_TEST',  # 11
              '../data//ECG200_TEST',  # 12
              '../data//ECGFiveDays_TEST',  # 13
              '../data//BeetleFly_TEST',  # 14
              '../data//BirdChicken_TEST',  # 15
              '../data//ItalyPowerDemand_TEST',  # 16
              '../data//SonyAIBORobotSurface_TEST',  # 17
              '../data//SonyAIBORobotSurfaceII_TEST',  # 18
              '../data//MiddlePhalanxOutlineAgeGroup_TEST',  # 19 (inverted dataset)
              '../data//MiddlePhalanxOutlineCorrect_TEST',  # 20 (inverted dataset)
              '../data//MiddlePhalanxTW_TEST',  # 21 (inverted dataset)
              '../data//ProximalPhalanxOutlineAgeGroup_TEST',  # 22
              '../data//ProximalPhalanxOutlineCorrect_TEST',  # 23
              '../data//ProximalPhalanxTW_TEST',  # 24 (inverted dataset)
              '../data//MoteStrain_TEST',  # 25
              '../data//MedicalImages_TEST',  # 26
              '../data//Strawberry_TEST',  # 27
              '../data//ToeSegmentation1_TEST',  # 28
              '../data//Coffee_TEST',  # 29
              '../data//Cricket_X_TEST',  # 30
              '../data//Cricket_Y_TEST',  # 31
              '../data//Cricket_Z_TEST',  # 32
              '../data//uWaveGestureLibrary_X_TEST',  # 33
              '../data//uWaveGestureLibrary_Y_TEST',  # 34
              '../data//uWaveGestureLibrary_Z_TEST',  # 35
              '../data//ToeSegmentation2_TEST',  # 36
              '../data//DiatomSizeReduction_TEST',  # 37
              '../data//Car_TEST',  # 38
              '../data//CBF_TEST',  # 39
              '../data//CinC_ECG_torso_TEST',  # 40
              '../data//Computers_TEST',  # 41
              '../data//Earthquakes_TEST',  # 42
              '../data//ECG5000_TEST',  # 43
              '../data//ElectricDevices_TEST',  # 44
              '../data//FaceAll_TEST',  # 45
              '../data//FaceFour_TEST',  # 46
              '../data//FacesUCR_TEST',  # 47
              '../data//Fish_TEST',  # 48
              '../data//FordA_TEST',  # 49
              '../data//FordB_TEST',  # 50
              '../data//Gun_Point_TEST',  # 51
              '../data//Ham_TEST',  # 52
              '../data//HandOutlines_TEST',  # 53
              '../data//Haptics_TEST',  # 54
              '../data//Herring_TEST',  # 55
              '../data//InlineSkate_TEST',  # 56
              '../data//LargeKitchenAppliances_TEST',  # 57
              '../data//Lighting2_TEST',  # 58
              '../data//Mallat_TEST',  # 59
              '../data//Meat_TEST',  # 60
              '../data//NonInvasiveFatalECG_Thorax1_TEST',  # 61
              '../data//NonInvasiveFatalECG_Thorax2_TEST',  # 62
              '../data//OliveOil_TEST',  # 63
              '../data//OSULeaf_TEST',  # 64
              '../data//PhalangesOutlinesCorrect_TEST',  # 65
              '../data//Phoneme_TEST',  # 66
              '../data//Plane_TEST',  # 67
              '../data//RefrigerationDevices_TEST',  # 68
              '../data//ScreenType_TEST',  # 69
              '../data//ShapeletSim_TEST',  # 70
              '../data//ShapesAll_TEST',  # 71
              '../data//SmallKitchenAppliances_TEST',  # 72
              '../data//StarLightCurves_TEST',  # 73
              '../data//SwedishLeaf_TEST',  # 74
              '../data//Symbols_TEST',  # 75
              '../data//synthetic_control_TEST',  # 76
              '../data//Trace_TEST',  # 77
              '../data//Two_Patterns_TEST',  # 78
              '../data//TwoLeadECG_TEST',  # 79
              '../data//UWaveGestureLibraryAll_TEST',  # 80
              '../data//Wafer_TEST',  # 81
              '../data//Worms_TEST',  # 82
              '../data//WormsTwoClass_TEST',  # 83
              '../data//Yoga_TEST',  # 84
              '../data//ACSF1_TEST',  # 85
              '../data//AllGestureWiimoteX_TEST',  # 86
              '../data//AllGestureWiimoteY_TEST',  # 87
              '../data//AllGestureWiimoteZ_TEST',  # 88
              '../data//BME_TEST',  # 89
              '../data//Chinatown_TEST',  # 90
              '../data//Crop_TEST',  # 91
              '../data//DodgerLoopDay_TEST',  # 92
              '../data//DodgerLoopGame_TEST',  # 93
              '../data//DodgerLoopWeekend_TEST',  # 94
              '../data//EOGHorizontalSignal_TEST',  # 95
              '../data//EOGVerticalSignal_TEST',  # 96
              '../data//EthanolLevel_TEST',  # 97
              '../data//FreezerRegularTrain_TEST',  # 98
              '../data//FreezerSmallTrain_TEST',  # 99
              '../data//Fungi_TEST',  # 100
              '../data//GestureMidAirD1_TEST',  # 101
              '../data//GestureMidAirD2_TEST',  # 102
              '../data//GestureMidAirD3_TEST',  # 103
              '../data//GesturePebbleZ1_TEST',  # 104
              '../data//GesturePebbleZ2_TEST',  # 105
              '../data//GunPointAgeSpan_TEST',  # 106
              '../data//GunPointMaleVersusFemale_TEST',  # 107
              '../data//GunPointOldVersusYoung_TEST',  # 108
              '../data//HouseTwenty_TEST',  # 109
              '../data//InsectEPGRegularTrain_TEST',  # 110
              '../data//InsectEPGSmallTrain_TEST',  # 111
              '../data//MelbournePedestrian_TEST',  # 112
              '../data//MixedShapesRegularTrain_TEST',  # 113
              '../data//MixedShapesSmallTrain_TEST',  # 114
              '../data//PickupGestureWiimoteZ_TEST',  # 115
              '../data//PigAirwayPressure_TEST',  # 116
              '../data//PigArtPressure_TEST',  # 117
              '../data//PigCVP_TEST',  # 118
              '../data//PLAID_TEST',  # 119
              '../data//PowerCons_TEST',  # 120
              '../data//Rock_TEST',  # 121
              '../data//SemgHandGenderCh2_TEST',  # 122
              '../data//SemgHandMovementCh2_TEST',  # 123
              '../data//SemgHandSubjectCh2_TEST',  # 124
              '../data//ShakeGestureWiimoteZ_TEST',  # 125
              '../data//SmoothSubspace_TEST',  # 126
              '../data//UMD_TEST',  # 127
              ]

MAX_SEQUENCE_LENGTH_LIST = [176,  # 0
                            251,  # 1
                            166,  # 2
                            256,  # 3
                            319,  # 4
                            234,  # 5
                            270,  # 6
                            270,  # 7
                            470,  # 8
                            80,  # 9
                            80,  # 10
                            80,  # 11
                            96,  # 12
                            136,  # 13
                            512,  # 14
                            512,  # 15
                            24,  # 16
                            70,  # 17
                            65,  # 18
                            80,  # 19
                            80,  # 20
                            80,  # 21
                            80,  # 22
                            80,  # 23
                            80,  # 24
                            84,  # 25
                            99,  # 26
                            235,  # 27
                            277,  # 28
                            286,  # 29
                            300,  # 30
                            300,  # 31
                            300,  # 32
                            315,  # 33
                            315,  # 34
                            315,  # 35
                            343,  # 36
                            345,  # 37
                            577,  # 38
                            128,  # 39
                            1639,  # 40
                            720,  # 41
                            512,  # 42
                            140,  # 43
                            96,  # 44
                            131,  # 45
                            350,  # 46
                            131,  # 47
                            463,  # 48
                            500,  # 49
                            500,  # 50
                            150,  # 51
                            431,  # 52
                            2709,  # 53
                            1092,  # 54
                            512,  # 55
                            1882,  # 56
                            720,  # 57
                            637,  # 58
                            1024,  # 59
                            448,  # 60
                            750,  # 61
                            750,  # 62
                            570,  # 63
                            427,  # 64
                            80,  # 65
                            1024,  # 66
                            144,  # 67
                            720,  # 68
                            720,  # 69
                            500,  # 70
                            512,  # 71
                            720,  # 72
                            1024,  # 73
                            128,  # 74
                            398,  # 75
                            60,  # 76
                            275,  # 77
                            128,  # 78
                            82,  # 79
                            945,  # 80
                            152,  # 81
                            900,  # 82
                            900,  # 83
                            426,  # 84
                            1460,  # 85
                            500,  # 86
                            500,  # 87
                            500,  # 88
                            128,  # 89
                            24,  # 90
                            46,  # 91
                            288,  # 92
                            288,  # 93
                            288,  # 94
                            1250,  # 95
                            1250,  # 96
                            1751,  # 97
                            301,  # 98
                            301,  # 99
                            201,  # 100
                            360,  # 101
                            360,  # 102
                            360,  # 103
                            455,  # 104
                            455,  # 105
                            150,  # 106
                            150,  # 107
                            150,  # 108
                            2000,  # 109
                            601,  # 110
                            601,  # 111
                            24,  # 112
                            1024,  # 113
                            1024,  # 114
                            361,  # 115
                            2000,  # 116
                            2000,  # 117
                            2000,  # 118
                            1345,  # 119
                            144,  # 120
                            2844,  # 121
                            1500,  # 122
                            1500,  # 123
                            1500,  # 124
                            385,  # 125
                            15,  # 126
                            150,  # 127
                            ]

NB_CLASSES_LIST = [37,  # 0
                   3,  # 1
                   3,  # 2
                   11,  # 3
                   7,  # 4
                   2,  # 5
                   25,  # 6
                   50,  # 7
                   5,  # 8
                   3,  # 9
                   2,  # 10
                   6,  # 11
                   2,  # 12
                   2,  # 13
                   2,  # 14
                   2,  # 15
                   2,  # 16
                   2,  # 17
                   2,  # 18
                   3,  # 19
                   2,  # 20
                   6,  # 21
                   3,  # 22
                   2,  # 23
                   6,  # 24
                   2,  # 25
                   10,  # 26
                   2,  # 27
                   2,  # 28
                   2,  # 29
                   12,  # 30
                   12,  # 31
                   12,  # 32
                   8,  # 33
                   8,  # 34
                   8,  # 35
                   2,  # 36
                   4,  # 37
                   4,  # 38
                   3,  # 39
                   4,  # 40
                   2,  # 41
                   2,  # 42
                   5,  # 43
                   7,  # 44
                   14,  # 45
                   4,  # 46
                   14,  # 47
                   7,  # 48
                   2,  # 49
                   2,  # 50
                   2,  # 51
                   2,  # 52
                   2,  # 53
                   5,  # 54
                   2,  # 55
                   7,  # 56
                   3,  # 57
                   2,  # 58
                   8,  # 59
                   3,  # 60
                   42,  # 61
                   42,  # 62
                   4,  # 63
                   6,  # 64
                   2,  # 65
                   39,  # 66
                   7,  # 67
                   3,  # 68
                   3,  # 69
                   2,  # 70
                   60,  # 71
                   3,  # 72
                   3,  # 73
                   15,  # 74
                   6,  # 75
                   6,  # 76
                   4,  # 77
                   4,  # 78
                   2,  # 79
                   8,  # 80
                   2,  # 81
                   5,  # 82
                   2,  # 83
                   2,  # 84
                   10,  # 85
                   10,  # 86
                   10,  # 87
                   10,  # 88
                   3,  # 89
                   2,  # 90
                   24,  # 91
                   7,  # 92
                   2,  # 93
                   2,  # 94
                   12,  # 95
                   12,  # 96
                   4,  # 97
                   2,  # 98
                   2,  # 99
                   18,  # 100
                   26,  # 101
                   26,  # 102
                   26,  # 103
                   6,  # 104
                   6,  # 105
                   2,  # 106
                   2,  # 107
                   2,  # 108
                   2,  # 109
                   3,  # 110
                   3,  # 111
                   10,  # 112
                   5,  # 113
                   5,  # 114
                   10,  # 115
                   52,  # 116
                   52,  # 117
                   52,  # 118
                   11,  # 119
                   2,  # 120
                   4,  # 121
                   2,  # 122
                   6,  # 123
                   5,  # 124
                   10,  # 125
                   3,  # 126
                   3,  # 127
                   ]
dataset_map = {'Adiac': 0,
               'ArrowHead': 1,
               'ChlorineConcentration': 2,
               'InsectWingbeatSound': 3,
               'Lightning7': 4,
               'Wine': 5,
               'WordSynonyms': 6,
               'FiftyWords': 7,
               'Beef': 8,
               'DistalPhalanxOutlineAgeGroup': 9,
               'DistalPhalanxOutlineCorrect': 10,
               'DistalPhalanxTW': 11,
               'ECG200': 12,
               'ECGFiveDays': 13,
               'BeetleFly': 14,
               'BirdChicken': 15,
               'ItalyPowerDemand': 16,
               'SonyAIBORobotSurface1': 17,
               'SonyAIBORobotSurface2': 18,
               'MiddlePhalanxOutlineAgeGroup': 19,
               'MiddlePhalanxOutlineCorrect': 20,
               'MiddlePhalanxTW': 21,
               'ProximalPhalanxOutlineAgeGroup': 22,
               'ProximalPhalanxOutlineCorrect': 23,
               'ProximalPhalanxTW': 24,
               'MoteStrain': 25,
               'MedicalImages': 26,
               'Strawberry': 27,
               'ToeSegmentation1': 28,
               'Coffee': 29,
               'CricketX': 30,
               'CricketY': 31,
               'CricketZ': 32,
               'UWaveGestureLibraryX': 33,
               'UWaveGestureLibraryY': 34,
               'UWaveGestureLibraryZ': 35,
               'ToeSegmentation2': 36,
               'DiatomSizeReduction': 37,
               'Car': 38,
               'CBF': 39,
               'CinCECGTorso': 40,
               'Computers': 41,
               'Earthquakes': 42,
               'ECG5000': 43,
               'ElectricDevices': 44,
               'FaceAll': 45,
               'FaceFour': 46,
               'FacesUCR': 47,
               'Fish': 48,
               'FordA': 49,
               'FordB': 50,
               'GunPoint': 51,
               'Ham': 52,
               'HandOutlines': 53,
               'Haptics': 54,
               'Herring': 55,
               'InlineSkate': 56,
               'LargeKitchenAppliances': 57,
               'Lightning2': 58,
               'Mallat': 59,
               'Meat': 60,
               'NonInvasiveFetalECGThorax1': 61,
               'NonInvasiveFetalECGThorax2': 62,
               'OliveOil': 63,
               'OSULeaf': 64,
               'PhalangesOutlinesCorrect': 65,
               'Phoneme': 66,
               'Plane': 67,
               'RefrigerationDevices': 68,
               'ScreenType': 69,
               'ShapeletSim': 70,
               'ShapesAll': 71,
               'SmallKitchenAppliances': 72,
               'StarLightCurves': 73,
               'SwedishLeaf': 74,
               'Symbols': 75,
               'SyntheticControl': 76,
               'Trace': 77,
               'TwoPatterns': 78,
               'TwoLeadECG': 79,
               'UWaveGestureLibraryAll': 80,
               'Wafer': 81,
               'Worms': 82,
               'WormsTwoClass': 83,
               'Yoga': 84,
               'ACSF1': 85,
               'AllGestureWiimoteX': 86,
               'AllGestureWiimoteY': 87,
               'AllGestureWiimoteZ': 88,
               'BME': 89,
               'Chinatown': 90,
               'Crop': 91,
               'DodgerLoopDay': 92,
               'DodgerLoopGame': 93,
               'DodgerLoopWeekend': 94,
               'EOGHorizontalSignal': 95,
               'EOGVerticalSignal': 96,
               'EthanolLevel': 97,
               'FreezerRegularTrain': 98,
               'FreezerSmallTrain': 99,
               'Fungi': 100,
               'GestureMidAirD1': 101,
               'GestureMidAirD2': 102,
               'GestureMidAirD3': 103,
               'GesturePebbleZ1': 104,
               'GesturePebbleZ2': 105,
               'GunPointAgeSpan': 106,
               'GunPointMaleVersusFemale': 107,
               'GunPointOldVersusYoung': 108,
               'HouseTwenty': 109,
               'InsectEPGRegularTrain': 110,
               'InsectEPGSmallTrain': 111,
               'MelbournePedestrian': 112,
               'MixedShapesRegularTrain': 113,
               # 'MixedShapesSmallTrain': 114,
               'PickupGestureWiimoteZ': 115,
               'PigAirwayPressure': 116,
               'PigArtPressure': 117,
               'PigCVP': 118,
               'PLAID': 119,
               'PowerCons': 120,
               'Rock': 121,
               'SemgHandGenderCh2': 122,
               'SemgHandMovementCh2': 123,
               'SemgHandSubjectCh2': 124,
               'ShakeGestureWiimoteZ': 125,
               'SmoothSubspace': 126,
               'UMD': 127
               }
