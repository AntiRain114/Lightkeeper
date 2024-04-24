import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'dart:typed_data';

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isDetecting = false;
  int _modelOutput = 0;
  late FlutterVision vision;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeVision();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras![0], ResolutionPreset.medium);
    await _controller!.initialize();
    setState(() {});
  }

  Future<void> _initializeVision() async {
    vision = FlutterVision();
    await vision.loadModel('assets/gestureModel_one.tflite');
  }

  Future<void> _detectGesture(CameraImage image) async {
    if (!_isDetecting) {
      _isDetecting = true;

      int startX = (image.width * 0.2).toInt();
      int endX = (image.width * 0.8).toInt();
      int startY = (image.height * 0.2).toInt();
      int endY = (image.height * 0.8).toInt();

      int rowStride = image.planes[0].bytesPerRow;
      int pixelStride = image.planes[0].bytesPerPixel!;

      int size = (endX - startX) * (endY - startY);
      var img = Uint8List(size);

      int index = 0;
      for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
          int pixelIndex = y * rowStride + x * pixelStride;
          img[index++] = image.planes[0].bytes[pixelIndex];
        }
      }

      img = _preprocessImage(img);

      var output = await vision.runModel(img);

      setState(() {
        _modelOutput = output[0].toInt();
      });

      _isDetecting = false;
    }
  }

  Uint8List _preprocessImage(Uint8List img) {
    int size = 100;
    var output = Uint8List(size * size);

    int rowStride = (img.length / size).round();
    for (int x = 0; x < size; x++) {
      for (int y = 0; y < size; y++) {
        int pixelIndex = y * rowStride + x;
        output[y * size + x] = img[pixelIndex];
      }
    }

    return output;
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Container();
    }

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Gesture Recognition')),
        body: Column(
          children: [
            Expanded(
              child: CameraPreview(_controller!),
            ),
            Text(
              'Predicted Gesture: $_modelOutput',
              style: TextStyle(fontSize: 20),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    vision.close();
    super.dispose();
  }
}