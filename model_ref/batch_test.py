from train_reformed import main

seeds = [51970489, 34110658, 74418540, 20436023, 93048409]
gpu = 0

for seed in seeds:
    main("pems08", gpu, seed)
    main("pems03", gpu, seed)
    main("pems04", gpu, seed)
    main("pemsbay", gpu, seed)
    main("metrla", gpu, seed)