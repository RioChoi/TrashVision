package com.rio.trashvision2

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.rio.trashvision2.ml.ConvertedModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import android.content.res.AssetFileDescriptor

import android.app.Activity
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.support.common.ops.NormalizeOp as NormalizeOp1

const val OPEN_GALLERY = 1 // 리퀘스트용 상수
private lateinit var photoImage: Bitmap

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val selectImage: Button = findViewById(R.id.selectImage)

        selectImage.setOnClickListener {
            loadImage()
        }
    }

    private fun loadImage() {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT

        startActivityForResult(Intent.createChooser(intent, "Load Picture"), OPEN_GALLERY)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == OPEN_GALLERY && resultCode == RESULT_OK) {
            try {
                val stream = data?.data?.let { contentResolver!!.openInputStream(it) }
                val imageView: ImageView = findViewById(R.id.imageView)
                val cardboardText: TextView = findViewById(R.id.cardboardText)

                if (::photoImage.isInitialized) photoImage.recycle()
                photoImage = BitmapFactory.decodeStream(stream)
                photoImage = Bitmap.createScaledBitmap(photoImage, 224, 224, false)
                imageView.setImageBitmap(photoImage)

                // First way
//                val outputs = TensorBuffer.createFixedSize( intArrayOf( 1 , 6 ) , DataType.FLOAT32 )
//                val image = TensorBuffer.createFixedSize( intArrayOf( 1 , 224 , 224 , 3 ) , DataType.UINT8 )
//
//                val imageProcessor = ImageProcessor.Builder()
//                    // Resize using Bilinear and Nearest Neighbor methods
//                    .add( ResizeOp( 224 , 224 , ResizeOp.ResizeMethod.BILINEAR ) )
//                    .build()
//
//                val tensorImage = TensorImage( DataType.UINT8 )
//                tensorImage.load( photoImage ) // PhotoImage : Bitmap
//                val processedImage = imageProcessor.process( tensorImage )
//
//                val imageBuffer = processedImage.buffer
//                val imageTensorBuffer = processedImage.tensorBuffer
//                val tensorProcessor = TensorProcessor.Builder()
//                    .add(NormalizeOp1(0, 255))
//                    .add( CastOp( DataType.FLOAT32 ) )
//                    .build()
//                val processedTensor = tensorProcessor.process( image )


                // Second way
                // Allocate ByteBuffer
                val photoImageByteArray = bitmapToByteArray(photoImage)
                val byteBuffer: ByteBuffer = ByteBuffer.allocate(1 *224 * 224 * 3 * 4)
                byteBuffer.put(photoImageByteArray)

                Toast.makeText(this, "$photoImageByteArray", Toast.LENGTH_SHORT).show()

                val model = ConvertedModel.newInstance(this)

                // Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(byteBuffer)

                // Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                cardboardText.text = outputFeature0.toString()

                // Releases model resources if no longer used.
                model.close()

            } catch (e: FileNotFoundException) {
                e.printStackTrace()
            }
        }
    }
}

fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
    val stream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
    return stream.toByteArray()
}


