package com.example.clip_executorch

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.util.Locale
import java.util.regex.Pattern

/**
 * CLIP Tokenizer (Full BPE Implementation)
 */
class Tokenizer(private val context: Context) {
    companion object {
        private const val TAG = "Tokenizer"
        const val MAX_SEQ_LENGTH = 77
        const val SOT_TOKEN = 49406
        const val EOT_TOKEN = 49407
        const val PAD_TOKEN = 0
        
        // CLIP pre-tokenization regex
        private val PATTERN = Pattern.compile("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]+|[^\\s\\p{L}\\p{N}]+", Pattern.CASE_INSENSITIVE)
    }

    private val vocab: Map<String, Int> by lazy {
        try {
            val jsonString = context.assets.open("vocab.json").bufferedReader().use { it.readText() }
            val json = JSONObject(jsonString)
            val map = mutableMapOf<String, Int>()
            val keys = json.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                map[key] = json.getInt(key)
            }
            map
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab.json", e)
            emptyMap()
        }
    }

    private val merges: List<Pair<String, String>> by lazy {
        try {
            val lines = context.assets.open("merges.txt").bufferedReader().use { it.readLines() }
            // CLIP merges file usually starts with a version header which we skip or it's not present
            lines.filter { it.contains(" ") && !it.startsWith("#") }.map {
                val parts = it.split(" ")
                Pair(parts[0], parts[1])
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load merges.txt", e)
            emptyList()
        }
    }

    // Rank of each merge for fast lookup
    private val bpeRanks: Map<Pair<String, String>, Int> by lazy {
        merges.withIndex().associate { it.value to it.index }
    }

    private val cache = mutableMapOf<String, String>()

    /**
     * Tokenize input text to CLIP format (77 tokens).
     */
    fun tokenize(text: String): IntArray {
        val result = mutableListOf<Int>()
        result.add(SOT_TOKEN)

        val matcher = PATTERN.matcher(text.lowercase(Locale.ROOT))
        while (matcher.find()) {
            val word = matcher.group()
            val bpeTokens = bpe(word)
            Log.d(TAG, "Word: '$word' -> BPE: '$bpeTokens'")
            for (token in bpeTokens.split(" ")) {
                val id = vocab[token]
                if (id != null) {
                    Log.d(TAG, "  Token: '$token' -> ID: $id")
                    result.add(id)
                } else {
                    Log.w(TAG, "  Token: '$token' NOT FOUND in vocab")
                }
            }
        }

        if (result.size > MAX_SEQ_LENGTH - 1) {
            val truncated = result.subList(0, MAX_SEQ_LENGTH - 1)
            truncated.add(EOT_TOKEN)
            return truncated.toIntArray()
        }

        result.add(EOT_TOKEN)
        val finalTokens = IntArray(MAX_SEQ_LENGTH) { PAD_TOKEN }
        for (i in result.indices) {
            finalTokens[i] = result[i]
        }
        return finalTokens
    }

    private fun bpe(token: String): String {
        if (cache.containsKey(token)) return cache[token]!!
        
        // CLIP adds </w> to the end of each word
        var word = token.map { it.toString() }.toMutableList()
        word[word.size - 1] = word.last() + "</w>"
        
        var pairs = getPairs(word)
        if (pairs.isEmpty()) return token + "</w>"

        while (true) {
            val bigram = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(bigram)) break

            val first = bigram.first
            val second = bigram.second
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                val subList = word.subList(i, word.size)
                val subIdx = subList.indexOf(first)
                val j = if (subIdx == -1) -1 else subIdx + i
                
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                i = j
                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }
        
        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        var prev = word[0]
        for (i in 1 until word.size) {
            val current = word[i]
            pairs.add(Pair(prev, current))
            prev = current
        }
        return pairs
    }
}
