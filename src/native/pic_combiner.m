//
//  pic_combiner.m
//  2D to 3D Studio - Native Apple Spatial HEIC Combiner
//
//  Synthesizes stereoscopic Left & Right images into Apple Spatial Photo (HEIC)
//  using Apple's native ImageIO and CoreGraphics frameworks.
//

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#import <math.h>

void printUsage(void) {
    printf("2D to 3D Studio - Native Spatial Photo Combiner\n");
    printf("Usage: run_picCombiner --left-image-path <left> --right-image-path <right> --output-image-path <out.heic>\n");
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSString *leftPath = nil;
        NSString *rightPath = nil;
        NSString *outputPath = nil;

        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            if (([arg isEqualToString:@"--left-image-path"] || [arg isEqualToString:@"-l"]) && i + 1 < argc) {
                leftPath = [NSString stringWithUTF8String:argv[++i]];
            } else if (([arg isEqualToString:@"--right-image-path"] || [arg isEqualToString:@"-r"]) && i + 1 < argc) {
                rightPath = [NSString stringWithUTF8String:argv[++i]];
            } else if (([arg isEqualToString:@"--output-image-path"] || [arg isEqualToString:@"-o"]) && i + 1 < argc) {
                outputPath = [NSString stringWithUTF8String:argv[++i]];
            }
        }

        if (!leftPath || !rightPath || !outputPath) {
            printUsage();
            return 1;
        }

        NSURL *leftURL = [NSURL fileURLWithPath:leftPath];
        NSURL *rightURL = [NSURL fileURLWithPath:rightPath];
        NSURL *outURL = [NSURL fileURLWithPath:outputPath];

        CGImageSourceRef leftSource = CGImageSourceCreateWithURL((__bridge CFURLRef)leftURL, NULL);
        if (!leftSource) {
            fprintf(stderr, "Error: Failed to read left image: %s\n", [leftPath UTF8String]);
            return 1;
        }
        CGImageRef leftImg = CGImageSourceCreateImageAtIndex(leftSource, 0, NULL);
        CFRelease(leftSource);
        if (!leftImg) {
            fprintf(stderr, "Error: Failed to decode left image: %s\n", [leftPath UTF8String]);
            return 1;
        }

        CGImageSourceRef rightSource = CGImageSourceCreateWithURL((__bridge CFURLRef)rightURL, NULL);
        if (!rightSource) {
            CGImageRelease(leftImg);
            fprintf(stderr, "Error: Failed to read right image: %s\n", [rightPath UTF8String]);
            return 1;
        }
        CGImageRef rightImg = CGImageSourceCreateImageAtIndex(rightSource, 0, NULL);
        CFRelease(rightSource);
        if (!rightImg) {
            CGImageRelease(leftImg);
            fprintf(stderr, "Error: Failed to decode right image: %s\n", [rightPath UTF8String]);
            return 1;
        }

        CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)outURL, CFSTR("public.heic"), 2, NULL);
        if (!dest) {
            CGImageRelease(leftImg);
            CGImageRelease(rightImg);
            fprintf(stderr, "Error: Failed to create HEIC image destination: %s\n", [outputPath UTF8String]);
            return 1;
        }

        CGFloat width = (CGFloat)CGImageGetWidth(leftImg);
        CGFloat height = (CGFloat)CGImageGetHeight(leftImg);
        CGFloat fovDegrees = 55.0;
        CGFloat fovRadians = fovDegrees * (M_PI / 180.0);
        CGFloat focalLength = 0.5 * width / tan(0.5 * fovRadians);

        NSArray *intrinsics = @[
            @(focalLength), @(0.0), @(width / 2.0),
            @(0.0), @(focalLength), @(height / 2.0),
            @(0.0), @(0.0), @(1.0)
        ];

        NSDictionary *properties = @{
            (id)kCGImagePropertyGroups: @{
                (id)kCGImagePropertyGroupIndex: @(0),
                (id)kCGImagePropertyGroupType: (id)kCGImagePropertyGroupTypeStereoPair,
                (id)kCGImagePropertyGroupImageIndexLeft: @(0),
                (id)kCGImagePropertyGroupImageIndexRight: @(1)
            },
            (id)kCGImagePropertyHEIFDictionary: @{
                @"CameraModel": @{
                    @"Intrinsics": intrinsics
                }
            }
        };

        CGImageDestinationAddImage(dest, leftImg, (__bridge CFDictionaryRef)properties);
        CGImageDestinationAddImage(dest, rightImg, (__bridge CFDictionaryRef)properties);

        bool success = CGImageDestinationFinalize(dest);
        CFRelease(dest);
        CGImageRelease(leftImg);
        CGImageRelease(rightImg);

        if (!success) {
            fprintf(stderr, "Error: Failed to finalize HEIC spatial photo.\n");
            return 1;
        }

        return 0;
    }
}
