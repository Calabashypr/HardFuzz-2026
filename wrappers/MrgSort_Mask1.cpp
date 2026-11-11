__aicore__ inline void MrgSort_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &tmp,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    int32_t ms_repeat = (data_length + 31) / 32;
    if (ms_repeat < 1) ms_repeat = 1;
    int32_t ms_src_elems = ms_repeat * 32;
    int32_t ms_sorted_elems = ms_repeat * 64;
    AscendC::LocalTensor<float> ms_src = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ms_src.SetSize(ms_src_elems * (int32_t)sizeof(float));
    offset_ptr += ms_src.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint32_t> ms_idx = tmp.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    ms_idx.SetSize(ms_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += ms_idx.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> ms_sorted = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ms_sorted.SetSize(ms_sorted_elems * (int32_t)sizeof(float));
    offset_ptr += ms_sorted.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> ms_dst = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ms_dst.SetSize(ms_sorted_elems * (int32_t)sizeof(float));
    offset_ptr += ms_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    for (int32_t i = 0; i < ms_src_elems; ++i) {
        ms_src.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
        ms_idx.SetValue(i, (uint32_t)i);
    }
    AscendC::Sort32<float>(ms_sorted, ms_src, ms_idx, ms_repeat);
    if (ms_repeat >= 4) {
        int32_t sub_len = ms_repeat * 32;
        AscendC::MrgSort4Info ms_params;
        ms_params.elementLengths[0] = (uint16_t)sub_len;
        ms_params.elementLengths[1] = (uint16_t)sub_len;
        ms_params.elementLengths[2] = (uint16_t)sub_len;
        ms_params.elementLengths[3] = (uint16_t)sub_len;
        ms_params.ifExhaustedSuspension = false;
        ms_params.validBit = 0b1111;
        ms_params.repeatTimes = 1;
        AscendC::MrgSortSrcList<float> ms_list;
        ms_list.src1 = ms_sorted[0];
        ms_list.src2 = ms_sorted[sub_len * 2];
        ms_list.src3 = ms_sorted[sub_len * 4];
        ms_list.src4 = ms_sorted[sub_len * 6];
        AscendC::MrgSort<float>(ms_dst, ms_list, ms_params);
        for (int32_t i = 0; i < data_length; ++i) {
            output.SetValue(i, ms_dst.GetValue(i * 2));
        }
    } else {
        for (int32_t i = 0; i < data_length; ++i) {
            output.SetValue(i, ms_sorted.GetValue(i * 2));
        }
    }
}
