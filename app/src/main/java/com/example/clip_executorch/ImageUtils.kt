package com.example.clip_executorch

import android.graphics.Bitmap
import android.util.Log

/**
 * Image preprocessing for CLIP model.
 * Expected input: 224x224, normalized.
 */
object ImageUtils {
    private const val TAG = "ImageUtils"
    const val TARGET_HEIGHT = 224
    const val TARGET_WIDTH = 224

    // CLIP Normalization constants
    private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    /**
     * Process bitmap to FloatArray in CHW format.
     */
    fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
        Log.d(TAG, "Input bitmap: ${bitmap.width}x${bitmap.height}, config=${bitmap.config}")
        
        // 1. Resize and Center Crop
        val scaledBitmap = resizeAndCenterCrop(bitmap)
        Log.d(TAG, "After resize/crop: ${scaledBitmap.width}x${scaledBitmap.height}")
        
        // 2. Extract pixels
        val pixels = IntArray(TARGET_WIDTH * TARGET_HEIGHT)
        scaledBitmap.getPixels(pixels, 0, TARGET_WIDTH, 0, 0, TARGET_WIDTH, TARGET_HEIGHT)
        
        // Diagnostic: check if all pixels are the same (blank image)
        val uniquePixels = pixels.toSet().size
        Log.d(TAG, "Unique pixel values: $uniquePixels")
        if (uniquePixels <= 3) {
            Log.w(TAG, "WARNING: Image has only $uniquePixels unique pixel values — may be blank/solid!")
        }
        
        // 3. Normalize and convert to CHW
        val floatData = FloatArray(3 * TARGET_WIDTH * TARGET_HEIGHT)
        for (i in 0 until TARGET_WIDTH * TARGET_HEIGHT) {
            val color = pixels[i]
            val r = ((color shr 16) and 0xFF) / 255.0f
            val g = ((color shr 8) and 0xFF) / 255.0f
            val b = (color and 0xFF) / 255.0f
            
            // Red channel
            floatData[i] = (r - MEAN[0]) / STD[0]
            // Green channel
            floatData[TARGET_WIDTH * TARGET_HEIGHT + i] = (g - MEAN[1]) / STD[1]
            // Blue channel
            floatData[2 * TARGET_WIDTH * TARGET_HEIGHT + i] = (b - MEAN[2]) / STD[2]
        }
        
        // Diagnostic: check float array statistics
        val rChannel = floatData.slice(0 until TARGET_WIDTH * TARGET_HEIGHT)
        val gChannel = floatData.slice(TARGET_WIDTH * TARGET_HEIGHT until 2 * TARGET_WIDTH * TARGET_HEIGHT)
        val bChannel = floatData.slice(2 * TARGET_WIDTH * TARGET_HEIGHT until 3 * TARGET_WIDTH * TARGET_HEIGHT)
        Log.d(TAG, "R channel: mean=${rChannel.average()}, min=${rChannel.minOrNull()}, max=${rChannel.maxOrNull()}")
        Log.d(TAG, "G channel: mean=${gChannel.average()}, min=${gChannel.minOrNull()}, max=${gChannel.maxOrNull()}")
        Log.d(TAG, "B channel: mean=${bChannel.average()}, min=${bChannel.minOrNull()}, max=${bChannel.maxOrNull()}")
        
        return floatData
    }

    private fun resizeAndCenterCrop(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        val scale = TARGET_WIDTH.toFloat() / Math.min(width, height)
        val scaledWidth = (width * scale).toInt()
        val scaledHeight = (height * scale).toInt()
        
        val resized = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        
        val x = (scaledWidth - TARGET_WIDTH) / 2
        val y = (scaledHeight - TARGET_HEIGHT) / 2
        
        return Bitmap.createBitmap(resized, x.coerceAtLeast(0), y.coerceAtLeast(0), TARGET_WIDTH, TARGET_HEIGHT)
    }
}
