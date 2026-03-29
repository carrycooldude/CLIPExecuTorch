package com.example.clip_executorch

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import com.facebook.soloader.SoLoader
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.foundation.clickable
import androidx.compose.foundation.shape.CircleShape

data class SearchResult(val uri: Uri, val bitmap: Bitmap, val similarity: Float, val embedding: FloatArray)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize SoLoader for Qualcomm Backend
        try {
            SoLoader.init(this, false)
            Log.i("CLIP", "SoLoader initialized")
            // Load the QNN backend delegate
            System.loadLibrary("qnn_executorch_backend")
            Log.i("CLIP", "Qualcomm backend library loaded")
        } catch (e: Exception) {
            Log.e("CLIP", "Failed to initialize Qualcomm backend", e)
        }

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFFBB86FC),
                    secondary = Color(0xFF03DAC6),
                    background = Color(0xFF121212),
                    surface = Color(0xFF1E1E1E)
                )
            ) {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    CLIPSearchScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CLIPSearchScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val clipModule = remember { ClipModule(context) }
    
    var isLoaded by remember { mutableStateOf(false) }
    var query by remember { mutableStateOf("") }
    
    // Single image search states
    var currentBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var currentEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var singleSimilarity by remember { mutableStateOf<Float?>(null) }
    
    // Multi-image search states
    var selectedResults by remember { mutableStateOf<List<SearchResult>>(emptyList()) }
    var isGallerySearch by remember { mutableStateOf(false) }
    
    var isLoading by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Ready") }

    // Launcher for camera
    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            currentBitmap = bitmap
            isGallerySearch = false
            selectedResults = emptyList()
            singleSimilarity = null
            scope.launch {
                isLoading = true
                statusMessage = "Encoding photo..."
                currentEmbedding = withContext(Dispatchers.IO) {
                    clipModule.encodeImage(bitmap)
                }
                statusMessage = if (currentEmbedding != null) "Photo ready" else "Encoding failed"
                isLoading = false
            }
        }
    }

    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            cameraLauncher.launch(null)
        } else {
            statusMessage = "Camera permission denied"
        }
    }

    // Launcher for gallery (multiple)
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            isGallerySearch = true
            currentBitmap = null
            currentEmbedding = null
            singleSimilarity = null
            scope.launch {
                isLoading = true
                statusMessage = "Encoding ${uris.size} photos..."
                val results = mutableListOf<SearchResult>()
                withContext(Dispatchers.IO) {
                    uris.forEachIndexed { index, uri ->
                        try {
                            statusMessage = "Encoding photo ${index + 1}/${uris.size}"
                            val inputStream = context.contentResolver.openInputStream(uri)
                            val bitmap = BitmapFactory.decodeStream(inputStream)
                            if (bitmap != null) {
                                val embedding = clipModule.encodeImage(bitmap)
                                if (embedding != null) {
                                    // Store with initial similarity 0
                                    results.add(SearchResult(uri, bitmap, 0f, embedding))
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("CLIP", "Error loading uri: $uri", e)
                        }
                    }
                }
                selectedResults = results
                statusMessage = "${results.size} photos ready"
                isLoading = false
            }
        }
    }

    // Auto-load models... (keep exist logic)
    LaunchedEffect(Unit) {
        if (!isLoaded && !isLoading) {
            scope.launch {
                isLoading = true
                statusMessage = "Preparing AI..."
                val visionFile = File(context.filesDir, "clip_vision.pte")
                val textFile = File(context.filesDir, "clip_text.pte")
                val success = withContext(Dispatchers.IO) {
                    // Copy assets to internal storage if needed
                    if (!visionFile.exists()) {
                        context.assets.open("models/clip_vision.pte").use { input ->
                            visionFile.outputStream().use { output -> input.copyTo(output) }
                        }
                    }
                    if (!textFile.exists()) {
                        context.assets.open("models/clip_text.pte").use { input ->
                            textFile.outputStream().use { output -> input.copyTo(output) }
                        }
                    }
                    // Load models on IO thread — XNNPACK loading is slow (~300MB), MUST NOT block UI
                    clipModule.loadModels(visionFile.absolutePath, textFile.absolutePath)
                }
                isLoaded = success
                isLoading = false
                statusMessage = if (success) "Neural engine ready" else "Model loading failed"
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(24.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "CLIP Search",
                    fontSize = 32.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = MaterialTheme.colorScheme.primary,
                    letterSpacing = (-1).sp
                )
                Text(
                    text = "On-Device Multimodal Search",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)
                )
            }
            
            // Mode Selectors
            Row {
                IconButton(onClick = {
                    val permissionCheck = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    if (permissionCheck == PackageManager.PERMISSION_GRANTED) {
                        cameraLauncher.launch(null)
                    } else {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }) {
                    Icon(androidx.compose.material.icons.Icons.Filled.AddAPhoto, contentDescription = "Camera", tint = MaterialTheme.colorScheme.primary)
                }
                IconButton(onClick = { galleryLauncher.launch("image/*") }) {
                    Icon(androidx.compose.material.icons.Icons.Filled.PhotoLibrary, contentDescription = "Gallery", tint = MaterialTheme.colorScheme.primary)
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        
        // Status Row
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (isLoaded) Color.Green else Color.Red)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = statusMessage,
                fontSize = 12.sp,
                color = if (isLoaded) MaterialTheme.colorScheme.onBackground.copy(alpha = 0.8f) else Color.Red
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Large Image or Grid
        Box(modifier = Modifier.weight(1f)) {
            if (isGallerySearch && selectedResults.isNotEmpty()) {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    contentPadding = PaddingValues(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(selectedResults.sortedByDescending { it.similarity }) { item ->
                        ResultCard(item, clipModule.normalizeSimilarity(item.similarity))
                    }
                }
            } else if (currentBitmap != null) {
                Box(modifier = Modifier.fillMaxSize()) {
                    Card(
                        modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(32.dp)),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Image(
                            bitmap = currentBitmap!!.asImageBitmap(),
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    }
                    
                    singleSimilarity?.let { rawScore ->
                        val normalizedScore = clipModule.normalizeSimilarity(rawScore)
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(24.dp)
                                .clip(RoundedCornerShape(20.dp))
                                .background(MaterialTheme.colorScheme.primaryContainer)
                                .padding(horizontal = 20.dp, vertical = 10.dp)
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = String.format("Match: %.0f%%", normalizedScore * 100),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 18.sp
                                )
                                Text(
                                    text = String.format("Raw Score: %.3f", rawScore),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.6f),
                                    fontSize = 11.sp
                                )
                            }
                        }
                    }
                }
            } else {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        "Capture a photo or pick from gallery to start",
                        color = Color.Gray,
                        fontSize = 14.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Prompt Input
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            placeholder = { Text("Search for anything...") },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            singleLine = true,
            trailingIcon = {
                if (query.isNotEmpty() && !isLoading && (currentEmbedding != null || selectedResults.isNotEmpty())) {
                    IconButton(onClick = {
                        scope.launch {
                            isLoading = true
                            statusMessage = "Searching..."
                            val queryEmb = withContext(Dispatchers.IO) { clipModule.encodeText(query) }
                            if (queryEmb != null) {
                                if (isGallerySearch) {
                                    selectedResults = selectedResults.map {
                                        it.copy(similarity = clipModule.calculateSimilarity(queryEmb, it.embedding))
                                    }
                                } else {
                                    singleSimilarity = clipModule.calculateSimilarity(queryEmb, currentEmbedding!!)
                                }
                                statusMessage = "Search complete"
                            }
                            isLoading = false
                        }
                    }) {
                        Icon(androidx.compose.material.icons.Icons.Filled.Search, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        )
    }
}

@Composable
fun ResultCard(result: SearchResult, normalizedScore: Float) {
    Card(
        modifier = Modifier.fillMaxWidth().aspectRatio(1f),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Box {
            Image(
                bitmap = result.bitmap.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            
            // Score Badge
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(8.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color.Black.copy(alpha = 0.7f))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = String.format("%.0f%%", normalizedScore * 100),
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = String.format("%.3f", result.similarity),
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 8.sp
                    )
                }
            }
        }
    }
}

// End of MainActivity.kt

