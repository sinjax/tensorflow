/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

import java.util.Collections;
import java.util.Vector;

public class LaneNetActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final String MODEL_FILE = "file:///android_asset/lanenet/graph.pb";
  private static final String INPUT = "Placeholder";
  private static final String OUTPUT = "add";
  private static final float TEXT_SIZE_DIP = 12;
  private Bitmap rgbFrameBitmap;
  private Bitmap croppedBitmap;
  private static final int INPUT_HEIGHT = 320;
  private int sensorOrientation;
  private static final boolean MAINTAIN_ASPECT = false;
  private Bitmap textureCopyBitmap;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private TensorFlowInferenceInterface inferenceInterface;
  private int[] intValues;
  private float[] floatValues;
  private BorderedText borderedText;

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_lanes;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    
    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();
    sensorOrientation = screenOrientation + rotation;
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
    int inputWidth, inputHeight;
    float scaleFactor = (float) INPUT_HEIGHT / previewHeight;
    inputWidth = INPUT_HEIGHT;
    inputHeight = (int)(previewWidth * scaleFactor);
    croppedBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
    frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            inputWidth, inputHeight,
            sensorOrientation, MAINTAIN_ASPECT);
    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    intValues = new int[inputWidth * inputHeight];
    floatValues = new float[inputWidth * inputHeight * 3];

    addCallback(
            new OverlayView.DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                renderDebug(canvas);
              }
            });
  }

  private void renderDebug(Canvas canvas) {
    Bitmap texture = this.textureCopyBitmap;
    if (texture != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor =
                      Math.min(
                      (float) canvas.getWidth() / texture.getWidth(),
                      (float) canvas.getHeight() / texture.getHeight());
      matrix.postScale(scaleFactor, scaleFactor);
      canvas.drawBitmap(texture, matrix, new Paint());
    }
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                Bitmap lanes = drawLanes(croppedBitmap);
                textureCopyBitmap = Bitmap.createBitmap(lanes);
                requestRender();
                readyForNextImage();
              }
            });
  }

  private Bitmap drawLanes(Bitmap bitmap) {

    int width, height;
    width = bitmap.getWidth();
    height = bitmap.getHeight();

    bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
      floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
      floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
    }


    inferenceInterface.feed(INPUT, floatValues, width, height, 3);
    try{
      inferenceInterface.run(new String[] {OUTPUT}, false);
    } catch(Exception e) {
      throw e;
    }
    inferenceInterface.fetch(OUTPUT, floatValues);
    for (int i = 0; i < intValues.length; ++i) {
      intValues[i] =
              0xFF000000
                      | (((int) (floatValues[i * 3] * 255)) << 16)
                      | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                      | ((int) (floatValues[i * 3 + 2] * 255));
    }
    Bitmap grayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    grayBitmap.setPixels(intValues, 0, width, 0, 0, width, height);
    return grayBitmap;
  }
}
