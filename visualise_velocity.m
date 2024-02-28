function [binary_image] = visualise_velocity(segmentation_file_path, image_sequence_dir, velocities, vessels_of_interest)

    % Normalize velocities to range from 0 to 1
    minVelocity = min(velocities);
    maxVelocity = max(velocities);
    normalizedVelocities = (velocities - minVelocity) / (maxVelocity - minVelocity);

    colormapSize = 64; % Size of the colormap
    myColormap = jet(colormapSize); % Using MATLAB's jet colormap

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
    disp(size(bw_skel_clean));

    %Skeletonised Segmentation
    imwrite(bw_skel_clean, '/Users/lohithkonathala/iib_project/skeletonized_image.png');

    % Read the first image in the sequence (for STI)
    first_img = imread(sprintf('%s/frame_0000.pgm', image_sequence_dir));

    % Ensure the first image is in grayscale for visualization
    if size(first_img, 3) == 3
        first_img = rgb2gray(first_img);
    end

    % Convert the first image to an RGB image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Overlay the skeleton on the first image
    overlay_color = [255, 0, 0]; % Red color for overlay
    for k = 1:3
        first_img_rgb(:,:,k) = first_img_rgb(:,:,k) .* uint8(~bw_skel_clean) + uint8(bw_skel_clean) * overlay_color(k);
    end

    % Find branch points
    branch_points = bwmorph(bw_skel_clean,'branchpoints');
    edtimage = 2 * bwdist(~binary_image);

    %diameterImage has the diameter value on all points on skeleton
    diameterImage = edtimage .* double(bw_skel_clean);

    branch_points_dilate = imdilate(branch_points, strel('disk',1));

    end_points = bwmorph(bw_skel_clean,'endpoints');
    discon = bw_skel_clean & ~branch_points_dilate;

    % Read the first image in the sequence
    first_img = imread(sprintf('%s/frame_0000.pgm', image_sequence_dir));

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

    stats = regionprops(labeled,'Centroid','Perimeter');
    pathlength = [stats.Perimeter]/2; % Calculate the path lengths
    [sortedLengths, sortedIndices] = sort(pathlength, 'descend');

    % Initialize a new binary image for highlighted segments
    highlighted_segments = false(size(discon_mini)); % Initialize with the same size but all zeros

    % Iterate through each vessel segment
    for ii = 1:numel(stats)
        % Find the rank of the current segment
        rank = find(sortedIndices == ii);
        disp(rank) % You can remove this disp if you don't need it anymore

        % Check if the current segment's rank is in the vessels_of_interest
        if ismember(rank, vessels_of_interest)
            % Add this segment to the highlighted_segments image
            highlighted_segments = highlighted_segments | (labeled == ii); % Add only the current segment
        end
    end

    % Now highlighted_segments contains only the vessels of interest

    % Overlay only the highlighted segments
    highlighted_rgb = label2rgb(imdilate(highlighted_segments, strel('disk',1)), jet(max(highlighted_segments(:))), 'k', 'shuffle');
    overlayed_image_highlighted = imadd(first_img_rgb, uint8(highlighted_rgb)); % Only add highlighted

    % Display the overlayed image with annotations for highlighted segments only
    figure;
    imshow(overlayed_image_highlighted);
    hold on;

    % Add length and rank annotations to the figure for highlighted segments
    for ii = 1:numel(stats)
        % Find the rank of the current segment
        rank = find(sortedIndices == ii);

        % Check if the current segment's rank is in the vessels_of_interest
        if ismember(rank, vessels_of_interest)
            interestIndex = find(vessels_of_interest == rank);
            if ~isempty(interestIndex)
                % Correcting for potential multiple matches, just use first one
                interestIndex = interestIndex(1);
                % Create the annotation text (length and rank)
                annotationText = sprintf('* %0.1f pixels, Rank: %d', velocities(interestIndex), rank);
                % Add the text to the image
                text(stats(ii).Centroid(1), stats(ii).Centroid(2), annotationText, ...
                    'FontUnits', 'normalized', 'Color', [0 0 0]);
            end
        end
    end

    title("Isolated Vessel Segments of Interest");
    hold off;

    % Use uiwait to keep the figure open until manually closed
    %uiwait(gcf);

    % Initialize a new RGB image for velocity-colored segments
    colored_segments = uint8(zeros([size(discon_mini), 3]));  % Assuming 'discon_mini' is your binary image size

    % Border properties
    borderWidth = 4;  % Adjust the border width as needed

    for ii = 1:numel(stats)
        % Find the rank of the current segment
        rank = find(sortedIndices == ii);

        % Check if the current segment's rank is in the vessels_of_interest
        if ismember(rank, vessels_of_interest)
            % Find the corresponding velocity index
            interestIndex = find(vessels_of_interest == rank, 1);
            if ~isempty(interestIndex)
                % Pick the color based on normalized velocity
                colorIndex = max(1, round(normalizedVelocities(interestIndex) * (colormapSize - 1)) + 1);
                segmentColor = myColormap(colorIndex, :);

                % Extract the binary image of the current segment
                segmentImage = (labeled == ii);

                % Create a border for the segment using the same segment color
                se = strel('disk', borderWidth);
                segmentBorder = imdilate(segmentImage, se) & ~segmentImage;

                % Colorize the segment and the border
                for channel = 1:3
                    channelImage = colored_segments(:, :, channel);
                    % Fill in the segment
                    channelImage(segmentImage) = segmentColor(channel) * 255; % Scale color to 0-255
                    % Apply the same color to the border
                    channelImage(segmentBorder) = segmentColor(channel) * 255; % Scale color to 0-255
                    colored_segments(:, :, channel) = channelImage;
                end
            end
        end
    end

    % Overlay the colored vessel segments on the first image
    overlayed_colored_image = imadd(first_img_rgb, colored_segments);

    % Display the overlayed image
    figure;
    imshow(overlayed_colored_image);
    title('Vessel Segments Colored by Velocity');
    hold off;
    uiwait(gcf);
