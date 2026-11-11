__aicore__ inline void GetMrgSortResult_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &tmp,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    int32_t gmr_repeat = (data_length + 31) / 32;
    if (gmr_repeat < 1) gmr_repeat = 1;
    if (gmr_repeat < 4) {
        for (int32_t i = 0; i < data_length; ++i) { output.SetValue(i, (float)data_length); }
        return;
    }
    int32_t gmr_src_elems = gmr_repeat * 32;
    int32_t gmr_sorted_elems = gmr_repeat * 64;
    AscendC::LocalTensor<float> gmr_src = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    gmr_src.SetSize(gmr_src_elems * (int32_t)sizeof(float));
    offset_ptr += gmr_src.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint32_t> gmr_idx = tmp.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    gmr_idx.SetSize(gmr_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += gmr_idx.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> gmr_sorted = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    gmr_sorted.SetSize(gmr_sorted_elems * (int32_t)sizeof(float));
    offset_ptr += gmr_sorted.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> gmr_dst = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    gmr_dst.SetSize(gmr_sorted_elems * (int32_t)sizeof(float));
    offset_ptr += gmr_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    for (int32_t i = 0; i < gmr_src_elems; ++i) {
        gmr_src.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
        gmr_idx.SetValue(i, (uint32_t)i);
    }
    AscendC::Sort32<float>(gmr_sorted, gmr_src, gmr_idx, gmr_repeat);
    int32_t gmr_sub_len = gmr_repeat * 32;
    AscendC::MrgSort4Info gmr_params;
    gmr_params.elementLengths[0] = (uint16_t)gmr_sub_len;
    gmr_params.elementLengths[1] = (uint16_t)gmr_sub_len;
    gmr_params.elementLengths[2] = (uint16_t)gmr_sub_len;
    gmr_params.elementLengths[3] = (uint16_t)gmr_sub_len;
    gmr_params.ifExhaustedSuspension = true;
    gmr_params.validBit = 0b1111;
    gmr_params.repeatTimes = 1;
    AscendC::MrgSortSrcList<float> gmr_list;
    gmr_list.src1 = gmr_sorted[0];
    gmr_list.src2 = gmr_sorted[gmr_sub_len * 2];
    gmr_list.src3 = gmr_sorted[gmr_sub_len * 4];
    gmr_list.src4 = gmr_sorted[gmr_sub_len * 6];
    AscendC::MrgSort<float>(gmr_dst, gmr_list, gmr_params);
    uint16_t gmr_r1 = 0, gmr_r2 = 0, gmr_r3 = 0, gmr_r4 = 0;
    AscendC::GetMrgSortResult(gmr_r1, gmr_r2, gmr_r3, gmr_r4);
    uint16_t gmr_total = gmr_r1 + gmr_r2 + gmr_r3 + gmr_r4;
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)(int32_t)gmr_total);
    }
}
