
__aicore__ inline void BilinearInterpolation_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    constexpr int32_t kHalfPerBlock = 16;
    constexpr int32_t kDstElems = 64;
    int32_t bi_off = (offset_ptr + 31) & ~31;
    offset_ptr = bi_off;
    AscendC::LocalTensor<half> src0_half, src1_half, dst_half;
    AscendC::LocalTensor<uint32_t> src0_offset;
    src0_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    src0_half.SetSize(kHalfPerBlock * (int32_t)sizeof(half));
    offset_ptr += src0_half.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    src1_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    src1_half.SetSize(kHalfPerBlock * (int32_t)sizeof(half));
    offset_ptr += src1_half.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    dst_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    dst_half.SetSize(kDstElems * (int32_t)sizeof(half));
    offset_ptr += dst_half.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    src0_offset = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    src0_offset.SetSize(8 * (int32_t)sizeof(uint32_t));
    offset_ptr += src0_offset.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint8_t> bilinear_tmp = sharedTmpBuffer[offset_ptr];
    bilinear_tmp.SetSize(2048);
    offset_ptr += bilinear_tmp.GetSize();
    for (int32_t i = 0; i < kHalfPerBlock; ++i) {
        float v0 = (i < data_length) ? input0.GetValue(i) : 0.0f;
        float v1 = (data_length > 0) ? input1.GetValue(0) : 0.0f;
        src0_half.SetValue(i, (half)v0);
        src1_half.SetValue(i, (half)v1);
    }
    for (int32_t i = 0; i < 8; ++i) {
        src0_offset.SetValue(i, 0U);
    }
    const uint64_t mask = (uint64_t)kDstElems;
    uint8_t hRepeat = 1;
    bool repeatMode = false;
    uint16_t dstBlkStride = 1;
    uint16_t vROffset = 128;
    uint8_t vRepeat = 1;
    AscendC::BilinearInterpolation(dst_half, src0_half, src0_offset, src1_half,
        mask, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat, bilinear_tmp);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)dst_half.GetValue(i));
    }
}
