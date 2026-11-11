__aicore__ inline void LoadUnzipIndex(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR srcIndexGm,
        const int32_t data_length) {

    AscendC::GlobalTensor<int8_t> indexGlobal;
    indexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(srcIndexGm), 512U);
    AscendC::LoadUnzipIndexCal(indexGlobal, 1U);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
