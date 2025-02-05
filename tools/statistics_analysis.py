import math

# seed = [51970489, 34110658, 74418540, 20436023, 93048409]

metrla_rmse_3 = [4.97145, 4.95736, 4.99114, 4.98271, 4.96570]
metrla_rmse_6 = [5.93624, 5.92249, 5.96185, 5.93691, 5.93962]
metrla_rmse_12 = [6.95726, 6.93459, 6.96010, 6.95576, 7.03074]
metrla_mae_3 = [2.60467, 2.59794, 2.60900, 2.60471, 2.59515]
metrla_mae_6 = [2.93512, 2.92783, 2.94407, 2.93629, 2.93204]
metrla_mae_12 = [3.31251, 3.29980, 3.31590, 3.31222, 3.33121]
metrla_mape_3 = [6.64356, 6.59342, 6.67685, 6.64018, 6.60921]
metrla_mape_6 = [7.97078, 7.92660, 8.01118, 7.98662, 7.95053]
metrla_mape_12 = [9.57210, 9.44848, 9.55880, 9.57150, 9.63209]

pemsbay_rmse_3 = [2.69038, 2.67653, 2.70245, 2.68320, 2.67032]
pemsbay_rmse_6 = [3.56927, 3.55965, 3.56329, 3.56575, 3.54384]
pemsbay_rmse_12 = [4.25048, 4.24875, 4.24337, 4.25339, 4.20831]
pemsbay_mae_3 = [1.27707, 1.27367, 1.28446, 1.27249, 1.27544]
pemsbay_mae_6 = [1.57550, 1.57911, 1.58343, 1.57370, 1.57503]
pemsbay_mae_12 = [1.84003, 1.85035, 1.84528, 1.84964, 1.83728]
pemsbay_mape_3 = [2.67726, 2.65362, 2.66723, 2.65924, 2.66521]
pemsbay_mape_6 = [3.54773, 3.52458, 3.52725, 3.51326, 3.52685]
pemsbay_mape_12 = [4.35780, 4.34014, 4.33850, 4.32573, 4.32326]

pems03_rmse = [25.55619, 25.60327, 26.25783, 25.63522, 25.04641]
pems03_mae = [14.90358, 14.86923, 15.20052, 14.94963, 14.86607]
pems03_mape = [14.91428, 15.04357, 15.32343, 15.05410, 14.98265]

pems03_rmse_m2 = [25.56571, 25.55619, 26.03535, 25.68509, 25.30376]
pems03_mae_m2 = [14.88020, 14.93169, 15.08773, 15.01726, 15.04975]
pems03_mape_m2 = [14.96328, 15.19661, 15.54908, 15.20548, 15.17457]

pems03_rmse_m3 = [26.60934, 26.58119, 26.26814, 26.73826, 26.13566]
pems03_mae_m3 = [15.30679, 15.46641, 15.28263, 15.52351, 15.05974]
pems03_mape_m3 = [15.24936, 15.32001, 15.50132, 15.29081, 15.27224]

pems03_rmse_m4 = [25.75967, 25.28410, 25.12594, 24.99873, 25.07328]
pems03_mae_m4 = [15.31602, 15.03458, 15.15870, 14.92301, 14.95297]
pems03_mape_m4 = [15.31305, 15.32249, 15.17915, 15.05410, 15.15347]

pems04_rmse = [30.98810, 30.45162, 30.58266, 30.75440, 30.46560]
pems04_mae = [18.22676, 18.03919, 18.21487, 18.19921, 18.13550]
pems04_mape = [12.05180, 11.87303, 11.95866, 12.01864, 11.94383]

pems04_rmse_m2 = [31.01123, 30.57903, 30.71916, 30.67926, 30.58200]
pems04_mae_m2 = [18.32393, 18.18295, 18.41942, 18.29226, 18.25019]
pems04_mape_m2 = [12.10489, 11.95047, 12.03682, 12.11349, 12.01075]

pems04_rmse_m3 = [30.69096, 30.70224, 30.70917, 30.71721, 30.40799]
pems04_mae_m3 = [18.29387, 18.28959, 18.34712, 18.26741, 18.28230]
pems04_mape_m3 = [12.06755, 12.04319, 11.98255, 11.99330, 11.95276]

pems04_rmse_m4 = [30.54909, 30.19714, 29.84247, 30.81676, 30.57952]
pems04_mae_m4 = [18.19715, 18.16573, 18.07862, 18.27287, 18.22075]
pems04_mape_m4 = [12.06380, 11.99998, 11.84608, 12.05740, 12.01199]

pems07_rmse = [33.04759, 32.60991, 32.60846, 32.81812, 32.23296]
pems07_mae = [19.24286, 19.05233, 19.00058, 19.23252, 18.95317]
pems07_mape = [8.00378, 7.89637, 7.91762, 7.96980, 7.85703]

pems07_rmse_m2 = [33.07272, 32.67254, 33.05024, 32.73606, 32.53222]
pems07_mae_m2 = [19.58438, 19.42016, 19.75506, 19.41117, 19.33615]
pems07_mape_m2 = [8.42048, 8.29584, 8.48438, 8.40879, 8.01762]

pems07_rmse_m3 = [32.81317, 32.80004, 32.59361, 32.34248, 32.49314]
pems07_mae_m3 = [19.21888, 19.26480, 19.10277, 19.03278, 19.08247]
pems07_mape_m3 = [7.98916, 8.00726, 7.97137, 7.98035, 7.98972]

pems07_rmse_m4 = [32.79260, 32.57682, 32.99474, 33.25718, 32.62750]
pems07_mae_m4 = [19.20576, 19.08216, 19.26739, 19.68187, 19.16326]
pems07_mape_m4 = [8.02407, 8.01567, 8.01293, 8.12818, 7.99065]

pems08_rmse = [22.73391, 22.75857, 22.79761, 22.75078, 22.63809]
pems08_mae = [13.10852, 13.13775, 13.14351, 13.15329, 13.11672]
pems08_mape = [8.62643, 8.68003, 8.69453, 8.70257, 8.64415]

pems08_rmse_m2 = [22.67538, 22.77743, 22.65683, 22.79366, 22.65027]
pems08_mae_m2 = [13.22685, 13.21392, 13.16033, 13.24169, 13.16961]
pems08_mape_m2 = [8.74938, 8.73585, 8.72281, 8.74315, 8.65900]

pems08_rmse_m3 = [23.26260, 23.27475, 23.23843, 23.21200, 23.19709]
pems08_mae_m3 = [13.45684, 13.43815, 13.36200, 13.42568, 13.43942]
pems08_mape_m3 = [8.86818, 8.88414, 8.84735, 8.85866, 8.84390]

pems08_rmse_m4 = [23.48612, 23.23566, 23.30236, 23.34785, 23.29884]
pems08_mae_m4 = [13.51972, 13.43344, 13.46807, 13.45882, 13.44942]
pems08_mape_m4 = [8.87512, 8.79240, 8.84216, 8.83142, 8.79402]

def average(nums: list, d=2):
    aver = sum(nums) / len(nums)
    factor = 10 ** d
    return math.ceil(aver * factor) / factor

print(f"""
    metrla_rmse_3: {average(metrla_rmse_3, 2)},
    metrla_rmse_6: {average(metrla_rmse_6, 2)},
    metrla_rmse_12: {average(metrla_rmse_12, 2)},
    metrla_mae_3： {average(metrla_mae_3, 2)},
    metrla_mae_6： {average(metrla_mae_6, 2)},
    metrla_mae_12： {average(metrla_mae_12, 2)},
    metrla_mape_3： {average(metrla_mape_3, 2)},
    metrla_mape_6： {average(metrla_mape_6, 2)},
    metrla_mape_12： {average(metrla_mape_12, 2)},
    
    pemsbay_rmse_3: {average(pemsbay_rmse_3, 2)},
    pemsbay_rmse_6: {average(pemsbay_rmse_6, 2)},
    pemsbay_rmse_12: {average(pemsbay_rmse_12, 2)},
    pemsbay_mae_3： {average(pemsbay_mae_3, 2)},
    pemsbay_mae_6： {average(pemsbay_mae_6, 2)},
    pemsbay_mae_12： {average(pemsbay_mae_12, 2)},
    pemsbay_mape_3： {average(pemsbay_mape_3, 2)},
    pemsbay_mape_6： {average(pemsbay_mape_6, 2)},
    pemsbay_mape_12： {average(pemsbay_mape_12, 2)},
    
    pems03_rmse (DMRC): {average(pems03_rmse, 2)},
    pems03_mae (DMRC): {average(pems03_mae, 2)},
    pems03_mape (DMRC): {average(pems03_mape, 2)},
    
    pems03_rmse (M2): {average(pems03_rmse_m2, 2)},
    pems03_mae (M2): {average(pems03_mae_m2, 2)},
    pems03_mape (M2): {average(pems03_mape_m2, 2)},

    pems03_rmse (M3): {average(pems03_rmse_m3, 2)},
    pems03_mae (M3): {average(pems03_mae_m3, 2)},
    pems03_mape (M3): {average(pems03_mape_m3, 2)},

    pems03_rmse (M4): {average(pems03_rmse_m4, 2)},
    pems03_mae (M4): {average(pems03_mae_m4, 2)},
    pems03_mape (M4): {average(pems03_mape_m4, 2)},

    pems04_rmse (DMRC): {average(pems04_rmse, 2)},
    pems04_mae (DMRC): {average(pems04_mae, 2)},
    pems04_mape (DMRC): {average(pems04_mape, 2)},

    pems04_rmse (M2): {average(pems04_rmse_m2, 2)},
    pems04_mae (M2): {average(pems04_mae_m2, 2)},
    pems04_mape (M2): {average(pems04_mape_m2, 2)},
    
    pems04_rmse (M3): {average(pems04_rmse_m3, 2)},
    pems04_mae (M3): {average(pems04_mae_m3, 2)},
    pems04_mape (M3): {average(pems04_mape_m3, 2)},
    
    pems04_rmse (M4): {average(pems04_rmse_m4, 2)},
    pems04_mae (M4): {average(pems04_mae_m4, 2)},
    pems04_mape (M4): {average(pems04_mape_m4, 2)},

    pems07_rmse (DMRC): {average(pems07_rmse, 2)},
    pems07_mae (DMRC): {average(pems07_mae, 2)},
    pems07_mape (DMRC): {average(pems07_mape, 2)},
    
    pems07_rmse (M2): {average(pems07_rmse_m2, 2)},
    pems07_mae (M2): {average(pems07_mae_m2, 2)},
    pems07_mape (M2): {average(pems07_mape_m2, 2)},

    pems07_rmse (M3): {average(pems07_rmse_m3, 2)},
    pems07_mae (M3): {average(pems07_mae_m3, 2)},
    pems07_mape (M3): {average(pems07_mape_m3, 2)},

    pems07_rmse (M4): {average(pems07_rmse_m4, 2)},
    pems07_mae (M4): {average(pems07_mae_m4, 2)},
    pems07_mape (M4): {average(pems07_mape_m4, 2)},

    pems08_rmse (DMRC): {average(pems08_rmse, 2)},
    pems08_mae (DMRC): {average(pems08_mae, 2)},
    pems08_mape (DMRC): {average(pems08_mape, 2)},
    
    pems08_rmse (M2): {average(pems08_rmse_m2, 2)},
    pems08_mae (M2): {average(pems08_mae_m2, 2)},
    pems08_mape (M2): {average(pems08_mape_m2, 2)},

    pems08_rmse (M3): {average(pems08_rmse_m3, 2)},
    pems08_mae (M3): {average(pems08_mae_m3, 2)},
    pems08_mape (M3): {average(pems08_mape_m3, 2)},
    
    pems08_rmse (M4): {average(pems08_rmse_m4, 2)},
    pems08_mae (M4): {average(pems08_mae_m4, 2)},
    pems08_mape (M4): {average(pems08_mape_m4, 2)},
""")

