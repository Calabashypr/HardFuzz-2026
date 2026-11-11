__aicore__ inline void Gatherb_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));

    constexpr int32_t BLOCK_BYTES = 32;
    constexpr int32_t ELEMS_PER_BLOCK = BLOCK_BYTES / (int32_t)sizeof(uint32_t);

    int32_t num_blocks = (data_length * (int32_t)sizeof(float) + BLOCK_BYTES - 1) / BLOCK_BYTES;
    uint8_t repeatTime = (uint8_t)((num_blocks + 7) / 8);
    if (repeatTime < 1) repeatTime = 1;

    int32_t gb_off = (offset_ptr + 31) & ~31;
    offset_ptr = gb_off;
    AscendC::LocalTensor<uint32_t> gb_src = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    gb_src.SetSize((uint32_t)data_length * (uint32_t)sizeof(uint32_t));
    offset_ptr += (int32_t)data_length * (int32_t)sizeof(uint32_t);
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<uint32_t> gb_dst = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    gb_dst.SetSize((uint32_t)data_length * (uint32_t)sizeof(uint32_t));
    offset_ptr += (int32_t)data_length * (int32_t)sizeof(uint32_t);
    offset_ptr = (offset_ptr + 31) & ~31;

    int32_t offset_elems = 8 * (int32_t)repeatTime;
    AscendC::LocalTensor<uint32_t> gb_offset = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    gb_offset.SetSize((uint32_t)offset_elems * (uint32_t)sizeof(uint32_t));
    offset_ptr += offset_elems * (int32_t)sizeof(uint32_t);
    offset_ptr = (offset_ptr + 31) & ~31;

    for (int32_t i = 0; i < data_length; ++i) {
        float fv = input0.GetValue(i);
        uint32_t uv;
        __builtin_memcpy(&uv, &fv, sizeof(uint32_t));
        gb_src.SetValue(i, uv);
    }

    for (int32_t i = 0; i < offset_elems; ++i) {
        gb_offset.SetValue(i, (uint32_t)(i * BLOCK_BYTES));
    }

    AscendC::GatherRepeatParams gb_params{1, 8};

    AscendC::Gatherb<uint32_t>(gb_dst, gb_src, gb_offset, repeatTime, gb_params);

    for (int32_t i = 0; i < data_length; ++i) {
        uint32_t uv = gb_dst.GetValue(i);
        float fv;
        __builtin_memcpy(&fv, &uv, sizeof(float));
        output.SetValue(i, fv);
    }
}
