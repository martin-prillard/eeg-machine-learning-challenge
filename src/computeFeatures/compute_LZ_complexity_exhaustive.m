% X_train
load('/cal/homes/prillard/challenge/features/data_challenge_binary.mat')
n = 10178
X_train_LZ = 1:n;
for k=1:n
    [C, H, gs] = calc_lz_complexity(X_train(k,:)', 'exhaustive', true);
    X_train_LZ(k) = C;
    disp(k)
end

csvwrite('/cal/homes/prillard/challenge/X_train_LZ_exhaustive.csv', X_train_LZ);

% X_test
n = 10087
X_test_LZ = 1:n;
for k=1:n
    [C, H, gs] = calc_lz_complexity(X_test(k,:)', 'exhaustive', true);
    X_test_LZ(k) = C;
    disp(k)
end

csvwrite('/cal/homes/prillard/challenge/X_test_LZ_exhaustive.csv', X_test_LZ);