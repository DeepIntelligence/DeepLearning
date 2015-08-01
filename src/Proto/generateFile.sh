#!/bin/bash

protoc DeepLearning.proto --cpp_out=.
mv DeepLearning.pb.h ../../include
mv DeepLearning.pb.cc ../
