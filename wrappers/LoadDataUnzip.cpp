__aicore__ inline void LoadDataUnzip(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR srcWeGm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    AscendC::GlobalTensor<int8_t> weGlobal;
    weGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(srcWeGm), 512U);

    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int8_t> dstB1;
    dstB1.SetAddrWithOffset(sharedTmpBuffer, offset_ptr);
    dstB1.SetSize(512);
    offset_ptr += dstB1.GetSize();
    AscendC::LoadDataUnzip<int8_t>(dstB1, weGlobal);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
