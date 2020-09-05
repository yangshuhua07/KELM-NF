function b_K = compute_kernel(mx,b_Kbuf,b_w_train,b_w_train_size,b_w_test,b_w_test_size)
        if mx == 1,
            b_K = mex_pk_mm(b_Kbuf, ...
                b_w_train, b_w_train_size, ...
                b_w_test, b_w_test_size, ...
                1);
        else
            b_K = zeros(b_train_size, b_test_size);
            for n = 1 : b_train_size,
                x1 = b_w_train(1:b_w_train_size(n), n);
                K1 = b_Kbuf(x1,:);
                for m = 1 : b_test_size,
                    x2 = b_w_test(1:b_w_test_size(m), m);
                    K2 = K1(:, x2);
                    b_K(n,m) = sum(sum(K2));
                end
            end
        end
    end

