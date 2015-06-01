% X_train
load('/cal/homes/prillard/challenge/features/data_challenge_binary.mat')
n = 10178
X_train_LZ = 1:n;
for k=1:n
    [C, H, gs] = calc_lz_complexity(X_train(k,:)', 'primitive', true);
    X_train_LZ(k) = C;
    disp(k)
end

csvwrite('/cal/homes/prillard/challenge/X_train_LZ_primitive.csv', X_train_LZ);

% X_test
n = 10087
X_test_LZ = 1:n;
for k=1:n
    [C, H, gs] = calc_lz_complexity(X_test(k,:)', 'primitive', true);
    X_test_LZ(k) = C;
    disp(k)
end

csvwrite('/cal/homes/prillard/challenge/X_test_LZ_primitive.csv', X_test_LZ);