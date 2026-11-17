__aicore__ inline void Scatter_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t sc_off = (offset_ptr + 31) & ~31;
    offset_ptr = sc_off;

    AscendC::LocalTensor<float> sc_src = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    sc_src.SetSize((uint32_t)data_length);
    offset_ptr += (int32_t)data_length * (int32_t)sizeof(float);
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<float> sc_dst = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    sc_dst.SetSize((uint32_t)data_length);
    offset_ptr += (int32_t)data_length * (int32_t)sizeof(float);
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<uint32_t> sc_offset = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    sc_offset.SetSize((uint32_t)data_length);
    offset_ptr += (int32_t)data_length * (int32_t)sizeof(uint32_t);
    offset_ptr = (offset_ptr + 31) & ~31;

    for (int32_t i = 0; i < data_length; ++i) {
        sc_src.SetValue(i, input0.GetValue(i));
        sc_offset.SetValue(i, (uint32_t)(i * (int32_t)sizeof(float)));
    }

    AscendC::PipeBarrier<PIPE_ALL>();
#if __NPU_ARCH__ == 2201
    // Apply scalar scatter on C220.
    for (int32_t i = 0; i < data_length; ++i) {
        const uint32_t dst_index = sc_offset.GetValue(i) / (uint32_t)sizeof(float);
        sc_dst.SetValue(dst_index, sc_src.GetValue(i));
    }
#else
    AscendC::Scatter<float>(sc_dst, sc_src, sc_offset, (uint32_t)0, (uint32_t)data_length);
#endif
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, sc_dst.GetValue(i));
    }
}
