% X_train
load('/cal/homes/prillard/challenge/features/data_challenge.mat')
m =
tao =
meanperiod =
maxiter = 5


n = 10178
X_train_LZ = 1:n;
for k=1:n
    [C, H, gs] = lyarosenstein(X_train(k,:)', );
    X_train_LZ(k) = C;
end

csvwrite('/cal/homes/prillard/challenge/X_train_LZ_primitive.csv', X_train_LZ);

% X_test
n = 10087
X_test_LZ = 1:n;
for k=1:n
    [C, H, gs] = lyarosenstein(X_test(k,:)', );
    X_test_LZ(k) = C;
end

csvwrite('/cal/homes/prillard/challenge/X_test_LZ_primitive.csv', X_test_LZ);
