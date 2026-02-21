#include "AlgoNoiseOracle.hpp"

// =========================================================
// THE ORACLE PIPELINE FOR A SINGLE CHANNEL
// =========================================================
void Process_Channel_Oracle
(
    const float* RESTRICT plane, 
    float* RESTRICT covLUT, 
    const int32_t width, 
    const int32_t height, 
    float* RESTRICT workspace,
    const int32_t workspaceSizeFloats
)
{
    CACHE_ALIGN float D[16][16];
    Generate_DCT_Basis(D);
    
    OracleCandidate* candidates = reinterpret_cast<OracleCandidate*>(workspace);
    const int32_t max_candidates = workspaceSizeFloats / (sizeof(OracleCandidate) / sizeof(float));
    
    // Dynamic striding to strictly fit memory limits
    int32_t step = 2;
    while (((width / step) * (height / step)) > max_candidates) step++;
    
    // 1. EXTRACT BLOCKS AND SEARCH FOR SPARSITY (FLATNESS)
    int32_t num_candidates = 0;
    for (int32_t y = 0; y < height - 15; y += step) { 
        for (int32_t x = 0; x < width - 15; x += step) {
            
            if (!Is_Valid_Block(plane, x, y, width)) continue;
            
            OracleCandidate& cand = candidates[num_candidates];
            cand.x = x; 
            cand.y = y;
            
            float sum = 0.0f;
            for(int32_t i = 0; i < 4; ++i) {
                for(int32_t j = 0; j < 4; ++j) sum += plane[(y + i) * width + (x + j)];
            }
            cand.mean = sum / 16.0f;
            
            Forward_DCT_4x4(&plane[y * width + x], width, cand.dct, D);
            
            cand.sd = 1e15f;
            // Search forward window (r1=4 to r2=14)
            for(int32_t sy = y + 4; sy < y + 14 && sy < height - 3; ++sy) {
                for(int32_t sx = x + 4; sx < x + 14 && sx < width - 3; ++sx) {
                    float cand_dct[16];
                    Forward_DCT_4x4(&plane[sy * width + sx], width, cand_dct, D);
                    float sd = Calculate_Sparse_Distance(cand.dct, cand_dct);
                    if (sd < cand.sd) cand.sd = sd;
                }
            }
            num_candidates++;
        }
    }
    
    // 2. BUILD THE HISTOGRAM (BINNING)
    int32_t numBins = std::max(1, (width * height) / 42000);
    if (numBins > 64) numBins = 64; 

    std::sort(candidates, candidates + num_candidates, [](const OracleCandidate& a, const OracleCandidate& b) {
        return a.mean < b.mean;
    });

    CACHE_ALIGN BinInfo bins[64] = {};
    const float min_mean = candidates[0].mean;
    const float max_mean = candidates[num_candidates - 1].mean;
    const float bin_width = (max_mean - min_mean + 0.001f) / numBins;
    
    int32_t current_bin = 0;
    float current_boundary = min_mean + bin_width;
    bins[0].start_idx = 0;
    
    for(int32_t i = 0; i < num_candidates; ++i) {
        if (candidates[i].mean > current_boundary && current_bin < numBins - 1) {
            current_bin++;
            current_boundary += bin_width;
            bins[current_bin].start_idx = i;
        }
        bins[current_bin].count++;
    }

    // 3. STATISTICAL EVALUATION (MAD) PER BIN
    for(int32_t b = 0; b < numBins; ++b) {
        if (bins[b].count == 0) continue;
        
        std::sort(candidates + bins[b].start_idx, candidates + bins[b].start_idx + bins[b].count, [](const OracleCandidate& c1, const OracleCandidate& c2) {
            return c1.sd < c2.sd;
        });
        
        // Take the top 0.5% (percentile=0.005)
        int32_t k_blocks = std::max(1, static_cast<int32_t>(bins[b].count * 0.005f));
        if (k_blocks > 4000) k_blocks = 4000; 
        
        CACHE_ALIGN float mad_temp[4000];
        
        for (int32_t k = 0; k < 16; ++k) {
            for (int32_t i = 0; i < k_blocks; ++i) {
                mad_temp[i] = candidates[bins[b].start_idx + i].dct[k];
            }
            float hat_std = Compute_MAD(mad_temp, k_blocks);
            bins[b].stds[k] = hat_std * 1.791106f; // K factor for 4x4
        }
        
        float b_mean = 0.0f;
        for (int32_t i = 0; i < k_blocks; ++i) b_mean += candidates[bins[b].start_idx + i].mean;
        bins[b].mean_val = b_mean / k_blocks;
    }

    // 4. CURVE SMOOTHING
    for(int32_t k = 0; k < 16; ++k) {
        CACHE_ALIGN float temp_stds[64] = {0};
        for(int32_t b = 0; b < numBins; ++b) {
            if (bins[b].count == 0) continue;
            float sum = 0.0f; int32_t cnt = 0;
            for(int32_t j = std::max(0, b - 1); j <= std::min(numBins - 1, b + 1); ++j) {
                if (bins[j].count > 0) { sum += bins[j].stds[k]; cnt++; }
            }
            temp_stds[b] = sum / cnt;
        }
        for(int32_t b = 0; b < numBins; ++b) {
            if (bins[b].count > 0) bins[b].stds[k] = temp_stds[b];
        }
    }

    // 5. FREQUENCY TO SPATIAL COVARIANCE (D^T * M * D)
    CACHE_ALIGN float bin_covs[64][256] = {0};
    for(int32_t b = 0; b < numBins; ++b) {
        if (bins[b].count == 0) continue;
        for (int32_t p = 0; p < 16; ++p) {
            for (int32_t q = 0; q < 16; ++q) {
                float sum = 0.0f;
                for (int32_t k = 0; k < 16; ++k) {
                    sum += D[k][p] * D[k][q] * (bins[b].stds[k] * bins[b].stds[k]);
                }
                bin_covs[b][p * 16 + q] = sum;
            }
        }
    }

    // 6. INTERPOLATE COVARIANCE MATRICES TO THE 256-LEVEL LOOK UP TABLE
    for(int32_t intensity = 0; intensity < 256; ++intensity) {
        int32_t b_left = -1, b_right = -1;
        for(int32_t b = 0; b < numBins; ++b) {
            if (bins[b].count > 0) {
                if (bins[b].mean_val <= intensity) b_left = b;
                if (bins[b].mean_val >= intensity && b_right == -1) b_right = b;
            }
        }
        
        if (b_left == -1) b_left = b_right;
        if (b_right == -1) b_right = b_left;
        if (b_left == -1) continue; 
        
        float weight_right = 0.0f;
        if (b_left != b_right) {
            weight_right = (intensity - bins[b_left].mean_val) / (bins[b_right].mean_val - bins[b_left].mean_val);
        }
        
        for(int32_t i = 0; i < 256; ++i) {
            covLUT[intensity * 256 + i] = bin_covs[b_left][i] * (1.0f - weight_right) + bin_covs[b_right][i] * weight_right;
        }
    }
}

// =========================================================
// MAIN ENTRY POINT
// =========================================================
void Estimate_Noise_Covariances
(
    const MemHandler& mem,
    const int32_t width,
    const int32_t height,
    const AlgoControls& algoCtrl
)
{
    // The workspace is sized at 4 * (width * height) floats
    const int32_t workspaceSizeFloats = width * height * 4;

    Process_Channel_Oracle(mem.Y_planar, mem.NoiseCov_Y, width, height, mem.OracleWorkspace, workspaceSizeFloats);
    Process_Channel_Oracle(mem.U_planar, mem.NoiseCov_U, width, height, mem.OracleWorkspace, workspaceSizeFloats);
    Process_Channel_Oracle(mem.V_planar, mem.NoiseCov_V, width, height, mem.OracleWorkspace, workspaceSizeFloats);
}