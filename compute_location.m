 function [ijw_train, ijw_test] = compute_location(nWind,train_size,test_size,train_pos,test_pos,wind,cols,rows)
        
        ijw_train = zeros(nWind, train_size);
        i = train_pos(:,1); j = train_pos(:,2);
        iw_begin = max(i-wind, 1); iw_end = min(i+wind, rows);
        jw_begin = max(j-wind, 1); jw_end = min(j+wind, cols);
        for n = 1 : train_size,
            iw = iw_begin(n) : iw_end(n); jw = jw_begin(n) : jw_end(n);
            iw_size = length(iw); jw_size = length(jw);
            iw = repmat(iw', 1, jw_size); jw = repmat(jw, iw_size, 1);
            ijw = iw + (jw - 1) * rows;
            ijw_train(1:iw_size*jw_size, n) = ijw(:);
            w_train_size(n) = iw_size*jw_size;
        end
        ijw_test = zeros(nWind, test_size);
        i = test_pos(:,1); j = test_pos(:,2);
        iw_begin = max(i-wind, 1); iw_end = min(i+wind, rows);
        jw_begin = max(j-wind, 1); jw_end = min(j+wind, cols);
        for n = 1 : test_size,
            iw = iw_begin(n) : iw_end(n); jw = jw_begin(n) : jw_end(n);
            iw_size = length(iw); jw_size = length(jw);
            iw = repmat(iw', 1, jw_size); jw = repmat(jw, iw_size, 1);
            ijw = iw + (jw - 1) * rows;
            ijw_test(1:iw_size*jw_size, n) = ijw(:);
            w_test_size(n) = iw_size*jw_size;
        end
    end
