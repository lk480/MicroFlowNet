function [binary_image] = velocity_overlay(segmentation_file_path, image_sequence_dir)
    
    % Load Segmentation 
    if exist(segmentation_file_path, 'file') ~= 2
        error('File not found: %s', segmentation_file_path);
    end
    % Load the image
    img = imread(segmentation_file_path);
    % Convert to grayscale if it's not
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    % Convert to binary (optional, based on your requirement)
    binary_image = imbinarize(img);
    
    % Skeletonize Segmentation
    bw_skel = bwmorph(binary_image, 'skel', inf);
    bw_skel_clean = bwmorph(bw_skel, 'spur', 5);

    % Read the first image in the sequence (for STI)
    first_img = imread(sprintf('%s/00000.pgm', image_sequence_dir));

    % Ensure the first image is in grayscale for visualization
    if size(first_img, 3) == 3
        first_img = rgb2gray(first_img);
    end

    % Convert the first image to an RGB image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Overlay the modified skeleton on the first image
    overlay_color = [255, 0, 0]; % Red color for overlay

    for k = 1:3
        first_img_rgb(:,:,k) = first_img_rgb(:,:,k) .* uint8(~discon) + uint8(discon) * overlay_color(k);
    end


    % Reset first image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Parameters
    min_length = 80; % Minimum size threshold

    % Remove objects smaller than the minimum size
    discon_mini = bwareaopen(discon, min_length, 8);
    labeled = bwlabel(discon_mini,8);


    %%%% ISOLATED SEGMENTS OVERLAY (ANNOTATED) %%%%

    stats = regionprops(labeled,'Centroid','Perimeter');
    pathlength = [stats.Perimeter]/2; % Calculate the path lengths

    % Sort the segments by length in descending order and get their ranks
    [sortedLengths, sortedIndices] = sort(pathlength, 'descend');

    % Create the RGB image for the labeled vessel segments
    labeled_rgb = label2rgb(imdilate(labeled, strel('disk',1)), jet(max(labeled(:))), 'k', 'shuffle');

    % Overlay the labeled vessel segments on the first image
    overlayed_image = imadd(first_img_rgb, uint8(labeled_rgb));

    % Display the overlayed image with annotations
    figure;
    imshow(overlayed_image);
    hold on;

    stats = regionprops(labeled, 'Centroid', 'Perimeter');

    % Add length and rank annotations to the figure
    for ii = 1:numel(stats)
        % Find the rank of the current segment
        rank = find(sortedIndices == ii);
        
        % Create the annotation text (length and rank)
        annotationText = sprintf('* %0.1f pixels, Rank: %d', pathlength(ii), rank);

        % Add the text to the image
        text(stats(ii).Centroid(1), stats(ii).Centroid(2), annotationText, ...
            'FontUnits', 'normalized', 'Color', [0 0 0]);
    end
    title("Isolated Vessel Segments");
    hold off;

    % Use uiwait to keep the figure open until manually closed
    uiwait(gcf);
