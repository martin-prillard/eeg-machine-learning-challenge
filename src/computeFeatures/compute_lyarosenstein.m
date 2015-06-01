% X_train
load('/cal/homes/prillard/challenge/features/data_challenge_binary.mat')
part = '1'
start = 0
stop = 10178
X_train_lyarosenstein = 1:stop-start+1;
for k=start:stop
    d = lyarosenstein(X_train(k,:)', 30, 1, 100, 35);
    X_train_lyarosenstein(k-start+1) = d;
    k
end
csvwrite(strcat('/cal/homes/prillard/challenge/X_train_lyarosenstein_',part,'.csv'), X_train_lyarosenstein);

% X_test
n = 10087
X_test_lyarosenstein = 1:n;
for k=1:n
    d = lyarosenstein(X_test(k,:)', 30, 1, 100, 35);
    X_test_lyarosenstein(k) = d;
    disp(k)
end
csvwrite('/cal/homes/prillard/challenge/X_test_lyarosenstein.csv', X_test_lyarosenstein);