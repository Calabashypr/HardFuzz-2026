__aicore__ inline void GatherMask(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    uint8_t src1Pattern = 7;
    uint32_t mask = (uint32_t)data_length;
    uint64_t rsvdCnt = 0;

    AscendC::GatherMaskParams gmp{ 1, 1, 0, 0 };
    AscendC::printf("[DBG] GatherMask enter len=%d\n", data_length);
    AscendC::GatherMask<float>(output, input0, src1Pattern, true, mask, gmp, rsvdCnt);
    AscendC::printf("[DBG] GatherMask exit rsvdCnt=%llu out0=%f\n", (unsigned long long)rsvdCnt, output.GetValue(0));
}
