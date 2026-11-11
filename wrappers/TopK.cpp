__aicore__ inline void TopK(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    const int32_t TK_OUTTER = 1;
    const int32_t TK_INNER = 32;
    const int32_t TK_K = 1;
    const int32_t TK_K_PAD = 8;
    const float TK_NEG_INF = -3.4028235e+38f;

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> tk_dst_val = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tk_dst_val.SetSize((TK_OUTTER * TK_K_PAD) * (int32_t)sizeof(float));
    offset_ptr += tk_dst_val.GetSize();

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<int32_t> tk_dst_idx = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    tk_dst_idx.SetSize((TK_OUTTER * TK_K_PAD) * (int32_t)sizeof(int32_t));
    offset_ptr += tk_dst_idx.GetSize();

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> tk_src = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tk_src.SetSize((TK_OUTTER * TK_INNER) * (int32_t)sizeof(float));
    offset_ptr += tk_src.GetSize();

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<int32_t> tk_src_idx = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    tk_src_idx.SetSize((TK_OUTTER * TK_INNER) * (int32_t)sizeof(int32_t));
    offset_ptr += tk_src_idx.GetSize();

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<bool> tk_finish = sharedTmpBuffer.ReinterpretCast<bool>()[offset_ptr / (int32_t)sizeof(bool)];
    tk_finish.SetSize(TK_OUTTER * (int32_t)sizeof(bool));
    offset_ptr += tk_finish.GetSize();

    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint8_t> tk_tmp = sharedTmpBuffer[offset_ptr];
    tk_tmp.SetSize(4096);
    offset_ptr += tk_tmp.GetSize();

    int32_t n = data_length;
    if (n < 1) n = 1;
    if (n > TK_INNER) n = TK_INNER;
    for (int32_t i = 0; i < TK_INNER; ++i) {
        tk_src.SetValue(i, (i < n) ? input0.GetValue(i) : TK_NEG_INF);
        tk_src_idx.SetValue(i, i);
    }
    tk_finish.SetValue(0, false);

    AscendC::TopKInfo tk_info;
    tk_info.outter = TK_OUTTER;
    tk_info.inner = TK_INNER;
    tk_info.n = n;
    TopkTiling tk_tiling{};
    AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NSMALL>(
        tk_dst_val, tk_dst_idx, tk_src, tk_src_idx, tk_finish, tk_tmp, TK_K, tk_tiling, tk_info, true);

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, tk_dst_val.GetValue(i % TK_K_PAD));
    }
}
