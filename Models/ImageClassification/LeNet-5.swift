// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import Foundation
import Checkpoints

// Original Paper:
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of sigmoid activations.

public protocol ExportableLayer {
   public var nameMappings: [String: String] { get }
}

extension Dense: ExportableLayer {
   public var nameMappings: [String: String] { ["weight": "w", "bias": "b"] }
}

extension Conv2D: ExportableLayer {
    public var nameMappings: [String: String] { ["filter": "f"] }
}

extension Array: ExportableLayer {
    public var nameMappings: [String: String] { ["h": "h"] }
}




public struct LeNet: Layer {
    public var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    public var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    public var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    public var flatten = Flatten<Float>()
    public var fc1 = Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    public var fc2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    public var fc3 = Dense<Float>(inputSize: 84, outputSize: 10)

    public init() {}
    
    public init(checkpoint: URL) throws {
        self.init()
    }
    
    public func writeCheckpoint(to location: URL, name: String) throws {
        var tensors = [String: Tensor<Float>]()
                
        self.recursivelyObtainTensors(self,scope:"model", tensors: &tensors, separator: "/")
       
        print(tensors)
       // recursivelyObtainTensors(model, scope: "model", tensors: &tensors, separator: "/")

        let writer = CheckpointWriter(tensors: tensors)
        try writer.write(to: location, name: name)
       /*
        // Copy auxiliary files if they need to be in different location than current
        // local storage.
        if location != storage {
            try writeAuxiliary(to: location)
        }
 */
    }

    
    public func recursivelyObtainTensors(
        _ obj: Any, scope: String? = nil, tensors: inout [String: Tensor<Float>], separator: String
    ) {
        let m = Mirror(reflecting: obj)
        let nameMappings: [String: String]
        if let exportableLayer = obj as? ExportableLayer {
            nameMappings = exportableLayer.nameMappings
        } else {
            nameMappings = [:]
        }

        var repeatedLabels: [String: Int] = [:]
        func suffix(for label: String) -> String {
            if let currentSuffix = repeatedLabels[label] {
                repeatedLabels[label] = currentSuffix + 1
                return "\(currentSuffix + 1)"
            } else {
                repeatedLabels[label] = 0
                return "0"
            }
        }

        let hasSuffix = (m.children.first?.label == nil)

        var path = scope
        for child in m.children {
            let label = child.label ?? "h"

            if let remappedLabel = nameMappings[label] {
                let labelSuffix = hasSuffix ? suffix(for: remappedLabel) : ""
                let conditionalSeparator = remappedLabel == "" ? "" : separator

                path = (scope != nil ? scope! + conditionalSeparator : "") + remappedLabel + labelSuffix
                if let tensor = child.value as? Tensor<Float> {
                    tensors[path!] = tensor
                }
            }
            recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
        }
    }
    
    
    
    
    
    
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, fc1, fc2, fc3)
    }
}
