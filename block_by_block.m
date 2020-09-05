function [train_rc, con] = block_by_block(b, blo, r_block, c_block, rows, cols, train_pos)
con = 0;
br = r_block(b); bc = c_block(b);
br_begin = 1 + (br-1) * blo; br_end = min(rows, br*blo); if br_begin > br_end, con = 1; end
bc_begin = 1 + (bc-1) * blo; bc_end = min(cols, bc*blo); if bc_begin > bc_end, con = 1; end
train_rc = (train_pos(:,1) >= br_begin) & (train_pos(:,1) <= br_end) & (train_pos(:,2) >= bc_begin) & (train_pos(:,2) <= bc_end);
end
