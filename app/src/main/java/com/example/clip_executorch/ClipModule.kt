package com.example.clip_executorch

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt

class ClipModule(private val context: Context) {
    companion object {
        private const val TAG = "ClipModule"
    }

    private var visionModule: Module? = null
    private var textModule: Module? = null
    private val tokenizer = Tokenizer(context)

    fun loadModels(visionPath: String, textPath: String): Boolean {
        return try {
            visionModule = Module.load(visionPath)
            textModule = Module.load(textPath)
            Log.i(TAG, "CLIP models loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading models", e)
            false
        }
    }

    /**
     * L2-normalize a vector in-place and return it.
     */
    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vec) sumSq += v.toDouble() * v.toDouble()
        val norm = sqrt(sumSq).toFloat()
        Log.d(TAG, "L2 norm before normalization: $norm")
        if (norm > 1e-12f) {
            for (i in vec.indices) vec[i] /= norm
        } else {
            Log.w(TAG, "WARNING: embedding norm is near-zero ($norm), vector is effectively constant!")
        }
        return vec
    }

    /**
     * Generate embedding for an image.
     * The model returns raw (un-normalized) features; we normalize in Kotlin.
     */
    fun encodeImage(bitmap: Bitmap): FloatArray? {
        val module = visionModule ?: return null
        return try {
            val floatData = ImageUtils.bitmapToFloatArray(bitmap)
            val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 224, 224))
            Log.d(TAG, "Image input tensor shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "First 5 pixel values: ${floatData.take(5).joinToString(", ")}")
            
            val outputs = module.forward(EValue.from(inputTensor))
            if (outputs.isEmpty()) {
                Log.e(TAG, "Vision module forward returned empty outputs")
                return null
            }
            val outputTensor = outputs[0].toTensor()
            val raw = outputTensor.getDataAsFloatArray()
            
            // Diagnostic: check if raw embedding is constant
            val nonZero = raw.count { it != 0f }
            val unique = raw.toSet().size
            Log.i(TAG, "Raw image embedding: size=${raw.size}, nonZero=$nonZero, unique=$unique")
            Log.i(TAG, "Raw image stats: mean=${raw.average()}, min=${raw.minOrNull()}, max=${raw.maxOrNull()}")
            Log.i(TAG, "Raw image embedding (first 10): ${raw.take(10).joinToString(", ")}")
            
            if (unique <= 3) {
                Log.e(TAG, "CRITICAL: Image embedding has only $unique unique values — model may be returning constant output!")
            }
            
            // Normalize in Kotlin
            l2Normalize(raw)
        } catch (e: Exception) {
            Log.e(TAG, "Error encoding image", e)
            null
        }
    }

    /**
     * Generate embedding for text.
     * The model returns features; we normalize in Kotlin for safety.
     * Uses int64 (Long) tensors to match the existing exported model.
     */
    fun encodeText(text: String): FloatArray? {
        val module = textModule ?: return null
        return try {
            val tokens = tokenizer.tokenize(text)
            Log.i(TAG, "Tokenized '$text': ${tokens.take(10).joinToString(", ")}...")
            
            val attentionMask = IntArray(Tokenizer.MAX_SEQ_LENGTH) { if (it < tokens.size && tokens[it] != 0) 1 else 0 }
            val activeTokens = attentionMask.sum()
            Log.i(TAG, "Active tokens (non-padding): $activeTokens")
            
            // XNNPACK text model compilation specifically requires 32-bit INT tensors 
            // for sequence embedding indexes.
            val inputIdsTensor = Tensor.fromBlob(tokens, longArrayOf(1, 77))
            val attentionMaskTensor = Tensor.fromBlob(attentionMask, longArrayOf(1, 77))
            
            val outputs = try {
                module.forward(EValue.from(inputIdsTensor), EValue.from(attentionMaskTensor))
            } catch (e: Exception) {
                Log.e(TAG, "FATAL: ExecuTorch text module failed to execute forward()", e)
                return null
            }
            
            if (outputs == null || outputs.isEmpty()) {
                Log.e(TAG, "Text module forward returned empty outputs")
                return null
            }
            
            val outputTensor = outputs[0].toTensor()
            val raw = outputTensor.getDataAsFloatArray()
            
            // Diagnostic: check if raw embedding is constant
            val nonZero = raw.count { it != 0f }
            val unique = raw.toSet().size
            Log.i(TAG, "Raw text embedding: size=${raw.size}, nonZero=$nonZero, unique=$unique")
            Log.i(TAG, "Raw text stats: mean=${raw.average()}, min=${raw.minOrNull()}, max=${raw.maxOrNull()}")
            Log.i(TAG, "Raw text embedding (first 10): ${raw.take(10).joinToString(", ")}")
            
            if (unique <= 3) {
                Log.e(TAG, "CRITICAL: Text embedding has only $unique unique values — model may be returning constant output!")
            }
            
            // Normalize in Kotlin
            l2Normalize(raw)
        } catch (e: Exception) {
            Log.e(TAG, "Error encoding text", e)
            null
        }
    }

    /**
     * Calculate cosine similarity between two embeddings.
     * Full formula: (A · B) / (|A| × |B|)
     * Since we L2-normalize above, this should be ≈ dot product,
     * but we compute full cosine similarity defensively.
     */
    fun calculateSimilarity(emb1: FloatArray, emb2: FloatArray): Float {
        if (emb1.size != emb2.size) {
            Log.e(TAG, "Embedding size mismatch: ${emb1.size} vs ${emb2.size}")
            return 0f
        }
        
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in emb1.indices) {
            dotProduct += emb1[i].toDouble() * emb2[i].toDouble()
            normA += emb1[i].toDouble() * emb1[i].toDouble()
            normB += emb2[i].toDouble() * emb2[i].toDouble()
        }
        
        val denominator = sqrt(normA) * sqrt(normB)
        val similarity = if (denominator > 1e-12) {
            (dotProduct / denominator).toFloat().coerceIn(-1f, 1f)
        } else {
            Log.w(TAG, "WARNING: denominator near zero in cosine similarity")
            0f
        }
        
        Log.i(TAG, "Similarity: $similarity (dot=$dotProduct, normA=${sqrt(normA)}, normB=${sqrt(normB)})")
        return similarity
    }
}
