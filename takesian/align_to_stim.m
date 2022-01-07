function [block] = align_to_stim(block, sound_time_delay)
% DOCUMENTATION IN PROGRESS
% 
% This function pulls out the trial windows for each sound presentation
% and stores in block
% 
% Argument(s): 
%   block (struct)
%   sound_time_delay (array) list of times (s) to delay each trial start time by
% 
% Returns: 
%   block (struct)
% 
% Notes:
%
% Variables needed from block.setup:
% -Sound_Time - from define_sound
% -timestamp - from define_sound
% -F - from define_suite2p
% -Fneu - from define_suite2p
% -spks - from define_suite2p
%
%
% TODO:
% Search 'TODO'

%%  Skip this function if Suite2p and Bruker data are not available

if ~isfield(block, 'Sound_Time') || ismissing(block.setup.suite2p_path)
    disp('Skipping align to stim...');
    return
end

disp('Aligning to stim...');

%define sound window
setup = block.setup;
if setup.stim_protocol == 9
    [Sound_Time,block] = align_to_water(block); % no sound - this variable is the time that the mouse actually licks the uncued water
    block.Sound_Time = Sound_Time;
else
    Sound_Time = block.Sound_Time;
end

% Used for Maryse's behavior stim
if nargin == 2
    Sound_Time = Sound_Time + sound_time_delay;
    block.setup.constant.after_stim = 3;
    setup.constant.after_stim = 3;
end

%Accomodate multiplane data
if isfield(block, 'MultiplaneData')
    multiplaneData = true;
    nPlanes = setup.XML.nPlanes;
else
    multiplaneData = false;
    nPlanes = 1;
end

%Accommodate channel 2
if isfield(block, 'F_chan2')
    chan2_exists = true;
else
    chan2_exists = false;
end

%% ALIGN DATA

for n = 1:nPlanes
    
    if multiplaneData
        planeName = strcat('plane', num2str(n - 1));
        F = block.F.(planeName);
        Fneu = block.Fneu.(planeName);
        F7 = block.F7.(planeName);
        spks = block.spks.(planeName);
        timestamp = block.timestamp.(planeName);
        if chan2_exists
            F_chan2 = block.F_chan2.(planeName);
            Fneu_chan2 = block.Fneu_chan2.(planeName);
            F7_chan2 = block.F7_chan2.(planeName);
        end
    else
        F = block.F;
        Fneu = block.Fneu;
        F7 = block.F7;
        spks = block.spks;
        timestamp = block.timestamp;
        if chan2_exists
            F_chan2 = block.F_chan2;
            Fneu_chan2 = block.Fneu_chan2;
            F7_chan2 = block.F7_chan2;
        end
    end
    
    %Remove NaNs from MarkPoints data to have pseudo baseline (as close as possible)
    if isfield(block.setup, 'hasMarkPoints')
        idx = block.timestamp_idx;
        F = F(:,idx);
        Fneu = Fneu(:,idx);
        F7 = F7(:,idx);
        spks = spks(:,idx);
        timestamp = timestamp(idx);
        if chan2_exists
            F_chan2 = F_chan2(:,idx);
            Fneu_chan2 = Fneu_chan2(:,idx);
            F7_chan2 = F7_chan2(:,idx);
        end
    end
    
    %If no cells exist in block/plane
    if isempty(F)
        return
    end
     
    %Preallocate variables and trial length in frames
    %Align everything to closest frame to sound start
    baseline_inFrames = round(setup.constant.baseline_length*block.setup.framerate);
    after_inFrames = round(setup.constant.after_stim*block.setup.framerate);
    duration_inFrames = baseline_inFrames + after_inFrames;

    nanMat = nan(size(F,1),length(Sound_Time),duration_inFrames);
    F_stim = nanMat;
    Fneu_stim = nanMat;
    F7_stim = nanMat;
    spks_stim = nanMat;
    time_to_sound = nan(length(Sound_Time),1);
    if chan2_exists
        F_chan2_stim = nanMat;
        Fneu_chan2_stim = nanMat;
        F7_chan2_stim = nanMat;
    end

    % loop through each stim-presenation
    for time=1:length(Sound_Time)

        sound = Sound_Time(time);
        [~, closest_frame_sound] = min(abs(timestamp(:)-sound));
        time_to_sound(time) = timestamp(closest_frame_sound) - sound;
        A = closest_frame_sound - baseline_inFrames;
        B = closest_frame_sound + after_inFrames - 1;
        a = 1;
        b = duration_inFrames;

        % loop through each "iscell" to find the stim-aligned 1) raw
        % fluoresence 2) neuropil signal 3) neuropil-corrected floresence 4)
        % df/F for the neuropil corrected fluoresence 5) deconvolved spikes

        %If user-defined baseline is before the beginning of the block
        %recording, set A = 1 and the beginning of the block will be nan
        if A < 1
            a = abs(A) + 2;
            A = 1;
        end

        %If user-defined trial is longer than block recording, take portion
        %of trial up to the end of recording, the rest of the frames will be nan
        if B > size(F7,2)
            B = size(F7,2);
            b = length(A:B);
        end
        
        %Catch problems
        if A > B
            error('A should not be greater than B. Check timestamps')
        end
        
        % pull out the frames aligned to a stim (defined in frames)
        F_stim(:,time,a:b) =  F(:,A:B);
        Fneu_stim(:,time,a:b) = Fneu(:,A:B);
        F7_stim(:,time,a:b) = F7(:,A:B);
        spks_stim(:,time,a:b) = spks(:,A:B);
        
        if chan2_exists
            F_stim_chan2(:,time,a:b) =  F_chan2(:,A:B);
            Fneu_stim_chan2(:,time,a:b) = Fneu_chan2(:,A:B);
            F7_stim_chan2(:,time,a:b) = F7_chan2(:,A:B);
        end
    end
    
    %Compute df_f and zscore based on LOCAL baseline
    F7_baseline = F7_stim(:,:,1:baseline_inFrames);
    df_f = (F7_stim - mean(F7_baseline,3,'omitnan'))./mean(F7_baseline,3,'omitnan');
    zscore = (F7_stim - mean(F7_baseline,3,'omitnan'))./std(F7_baseline,[],3,'omitnan');

    if chan2_exists
        F7_baseline_chan2 = F7_stim_chan2(:,:,1:baseline_inFrames);
        df_f_chan2 = (F7_stim_chan2 - mean(F7_baseline_chan2,3,'omitnan'))./mean(F7_baseline_chan2,3,'omitnan');
        zscore_chan2 = (F7_stim_chan2 - mean(F7_baseline_chan2,3,'omitnan'))./std(F7_baseline_chan2,[],3,'omitnan');        
    end
    
    % SAVE ALIGNED DATA TO BLOCK
    if multiplaneData
         block.aligned_stim.F_stim.(planeName) = F_stim;
         block.aligned_stim.Fneu_stim.(planeName) = Fneu_stim;
         block.aligned_stim.F7_stim.(planeName) = F7_stim;
         block.aligned_stim.spks_stim.(planeName) = spks_stim;
         block.aligned_stim.df_f.(planeName) = df_f;
         block.aligned_stim.zscore.(planeName) = zscore;
         block.aligned_stim.time_to_sound.(planeName) = time_to_sound;
         if chan2_exists
             block.aligned_stim.F_stim_chan2.(planeName) = F_stim_chan2;
             block.aligned_stim.Fneu_stim_chan2.(planeName) = Fneu_stim_chan2;
             block.aligned_stim.F7_stim_chan2.(planeName) = F7_stim_chan2;
             block.aligned_stim.df_f_chan2.(planeName) = df_f_chan2;
             block.aligned_stim.zscore_chan2.(planeName) = zscore_chan2;
         end
    else
         block.aligned_stim.F_stim = F_stim;
         block.aligned_stim.Fneu_stim = Fneu_stim;
         block.aligned_stim.F7_stim = F7_stim;
         block.aligned_stim.spks_stim = spks_stim;
         block.aligned_stim.df_f = df_f;
         block.aligned_stim.zscore = zscore;
         block.aligned_stim.time_to_sound = time_to_sound;
         if chan2_exists
             block.aligned_stim.F_stim_chan2 = F_stim_chan2;
             block.aligned_stim.Fneu_stim_chan2 = Fneu_stim_chan2;
             block.aligned_stim.F7_stim_chan2 = F7_stim_chan2;
             block.aligned_stim.df_f_chan2 = df_f_chan2;
             block.aligned_stim.zscore_chan2 = zscore_chan2;
         end
    end
end

 
end