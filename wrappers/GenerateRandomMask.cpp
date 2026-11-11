
__aicore__ inline uint64_t GenerateRandomMask(const int32_t data_length) {
    uint64_t mask = 0;

    uint64_t seed = (uint64_t)data_length * 0x9e3779b97f4a7c15ULL;
    for (int32_t i = 0; i < data_length && i < 64; ++i) {

        uint64_t rand_val = (seed >> (i % 6)) ^ (seed << (i % 5));
        if ((rand_val & 0x3ULL) != 0ULL) {
            mask |= (1ULL << i);
        }
    }
    return mask;
}
