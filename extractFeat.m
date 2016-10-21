function [featMat, featureStr, userStats] = extractFeatures(touchfile)
%{

function [featMat, featureStr, userStats] = extractFeatures(touchfile)

This function takes a matrix of touch screen recordings as an input and
outputs a set of features for each stroke in the dataset.

This function was modified by Vineet Shenoy to include polar data

input:
    touchfile:  filename of the comma-separated file with the touch data.
                The order of the columns must be
                'phoneID', 'userID', 'documentID', 'time[ms]',
                'action_type', 'x-coord.', 'y-coord.'.
                action_ype can have three values 0: touch down
                                                 1: touch up
                                                 2: move
output:
    featMat:    A Nstroke times Nfeatures matrix.
    featureStr: Nfeatures cells, where cell i contains a descriptive string
                for feature i.
    userStats:  Number of strokes per user and per document (aka app);
                breakdown into left/right/up/down strokes. Stats will be
                prompted on screen: totals and direction breakdown
%}

%Putting in a simple comment to test git branch

%% constants and conversion factors

% raw feature columns
col_phoneID = 1;
col_user = 2;
col_doc  = 3;
col_time = 4;
col_act  = 5;
col_x = 7;
col_y = 8;
col_orient = 6;
col_press = 9;
col_area = 10;
col_Forient = 11;
col_rho = 12;
col_theta = 13;


% number of extracted features
Nfeat = 52;


% feature descriptors
featureStr{1} ='user id';
featureStr{2  } = 'doc id';
featureStr{3  } = 'inter-stroke time';
featureStr{4  } = 'stroke duration';
featureStr{5  } = 'start $x$';
featureStr{6  } = 'start $y$';
featureStr{7  } = 'stop $x$';
featureStr{8} = 'stop $y$';
featureStr{9} = 'direct end-to-end distance';
featureStr{10} = ' mean resultant lenght';
featureStr{11} = 'up/down/left/right flag';
featureStr{12} = 'direction of end-to-end line';
featureStr{13} = 'phone id';
featureStr{14} = '20\%-perc. pairwise velocity';
featureStr{15} = '50\%-perc. pairwise velocity';
featureStr{16} = '80\%-perc. pairwise velocity';
featureStr{17} = '20\%-perc. pairwise acc';
featureStr{18} = '50\%-perc. pairwise acc';
featureStr{19} = '80\%-perc. pairwise acc';
featureStr{20 } = 'median velocity at last 3 pts';
featureStr{21 } = 'largest deviation from end-to-end line';
featureStr{22} = '20\%-perc. dev. from end-to-end line';
featureStr{23} = '50\%-perc. dev. from end-to-end line';
featureStr{24} = '80\%-perc. dev. from end-to-end line';
featureStr{25  } = 'average direction';
featureStr{26 } = 'length of trajectory';
featureStr{27 } = 'ratio end-to-end dist and length of trajectory';
featureStr{28 } = 'average velocity';
featureStr{29} = 'median acceleration at first 5 points';
featureStr{30 } = 'mid-stroke pressure';
featureStr{31 } = 'mid-stroke area covered';
featureStr{32 } = 'mid-stroke finger orientation';
featureStr{33} = 'change of finger orientation';
featureStr{34} = 'phone orientation';
%>VS
featureStr{35} = 'rho start';
featureStr{36} = 'theta  start';
featureStr{37} = 'rho end';
featureStr{38} = 'theta end';
featureStr{39} = '20\% drho/dt';
featureStr{40} = '50\% drho/dt';
featureStr{41} = '80\% drho/dt';
featureStr{42} = '20\% dtheta/dt';
featureStr{43} = '50\% dtheta/dt';
featureStr{44} = '80\% dtheta/dt';
featureStr{45} = '20\% d^2rho/dt^2';
featureStr{46} = '50\% d^2rho/dt^2';
featureStr{47} = '80\% d^2rho/dt^2';
featureStr{48} = '20\% d^2theta/dt^2';
featureStr{49} = '50\% d^2theta/dt^2';
featureStr{50} = '80\% d^2theta/dt^2';
featureStr{51} = 'median drho/dt of last 3 points'
featureStr{52} = 'median dtheta/dt at last 3 points'
featureStr{53} = 'x-displacement'
featureStr{54} = 'y-displacement'
featureStr{55} = 'rho-displacement'
featureStr{56} = 'theta-displacement'
%<VS

% dots per inch of phones
% 2do: maybe convert from pixels to mm outside of this function and omit
% the following case distinction
pixTommFac(1) = 1/252; %Nexus one,
pixTommFac(2) = 1/233; %Nexus S,
pixTommFac(3) = 1/252; %Nexus one,
pixTommFac(4) = 1/233; %Samsung Galaxy S,
pixTommFac(5) = 1/252; %Droid Incredible,

%convert from inches to mm
pixTommFac = pixTommFac * 25.4;


% one feature (mean resultant length) requires the circular statistics
% toolbox:
% http://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics
% add your local path of the toolbox here after installation
addpath('..\..\..\shared_codeRep\CircStat2011f')



%% read raw data
t = dlmread(touchfile);



%%  preprocess
%convert ms to s
t(:,col_time) = t(:,col_time)/1000;

% flip sign of yaxis
t(:,col_y) = -t(:,col_y);

% start counting from 1. Adds value of one to every cell
%in this column
t(:,col_phoneID) = t(:,col_phoneID) +1;
t(:,col_user) = t(:,col_user)+1;
t(:,col_doc)  = t(:,col_doc)+1;



%% global statistics
indivPrctlVals = [20 50 80];

% find beginning of strokes
downInd = find( t(:,col_act)==0 );

% number of strokes
Nstrokes = length( downInd );
size(t,1)
downInd = [downInd ; size(t, 1)]; %putting the last row
% as the endpoint


% init feature output
featMat = NaN* zeros(Nstrokes, Nfeat);


disp(['extracting features of ' num2str( Nstrokes ) ' strokes..']);

%% extract features from each individual stroke
for i = 1:Nstrokes
    
    %% get stroke
    strokeInd = downInd(i):downInd(i+1)-1; %indices that are part of stroke
    x_stroke = t(strokeInd,:);  % Gets all data associated with certain measurement
    
    %dbstop in extractFeat at 176 if i>=457
    
    
    % sanity check
    if t(strokeInd(end),col_act)~=1
        disp(['last point of stroke ' num2str(i) ' is not "up"']);
        continue;
    end
    
    
    % number of measurements of stroke
    npoints = round(size(x_stroke,1));
    
    %% id stuff (do not use as features!)
    % user id
    featMat(i, 1) = x_stroke(1,col_user);
    
    % doc id
    featMat(i, 2) = x_stroke(1,col_doc);
    
    % phone id
    featMat(i, 13) = x_stroke(1,col_phoneID);
    
    
    %% features
    % convert from pixels to mm. (x,y) is now in millimeters
    x_stroke(:, [col_x col_y]) =  pixTommFac(featMat(i, 13)) * x_stroke(:, [col_x col_y]);
    
    %Obtain [theta,rho] data for stroke
    
    [theta, rho] = cart2pol(x_stroke(:,col_x),x_stroke(:,col_y));
    x_stroke(:, [col_theta col_rho]) = [theta,rho];
    
    % time to next stroke (0 if last stroke in dataset)
    % if last stroke in dset or last sroke of this user, set to Nan
    if featMat(i, 3) == 0 || t(downInd(i+1),col_user) ~= featMat(i, 1)
        featMat(i, 3) = NaN;
    else
        featMat(i, 3) = t(min([ downInd(Nstrokes) downInd(i+1)]),col_time) - x_stroke(1,col_time);
    end
    
    
    % time to last point of this stroke
    featMat(i, 4) =  x_stroke(end,col_time) - x_stroke(1,col_time);
    
    % x-pos start
    featMat(i, 5) = x_stroke(1,col_x);
    
    
    
    % y-pos start
    featMat(i, 6) = x_stroke(1,col_y);
    
    % x-pos end
    featMat(i, 7) = x_stroke(end,col_x);
    
    % y-pos end
    featMat(i, 8) = x_stroke(end,col_y);
    
    %>VS
    %x displacement
    featMat(i,53) = featMat(i,7) - featMat(i,5);
    
    %y displacment
    featMat(i,54) = featMat(i,8) - featMat(i,6);
    
    %rho start
    featMat(i,35) = x_stroke(1,col_rho);
    % theta start
    featMat(i,36) = x_stroke(1, col_theta);
    
    %rho end (final - initial)
    featMat(i,37) = x_stroke(end, col_rho);
    %theta end (final - initial)
    featMat(i,38) = x_stroke(end, col_theta);
    
    
    %rho displacement (final - initial)
    featMat(i,55) = featMat(i,37) - featMat(i,35);
    
    %theta displacement (final - initial) 
    featMat(i,56) = featMat(i,38) - featMat(i,36);
    
    %<VS
    % full dist
    featMat(i, 9) = sqrt((featMat(i, 8)-featMat(i, 6))^2 + (featMat(i, 7)-featMat(i, 5))^2);
    
    %% pairwise stuff
    
    % x-displacement
    xdispl = filter([1 -1] ,1 ,x_stroke(:,col_x));
    xdispl(1) = [];
    % y-displacement
    ydispl = filter([1 -1] ,1 ,x_stroke(:,col_y));
    ydispl(1) = [];
    
    %>VS
    %rho displacement
    rhodispl = filter([1 -1] ,1 ,x_stroke(:,col_rho));
    rhodispl(1) = [];
    
    %theta displacement
    thetadispl = filter([1 -1] ,1 ,x_stroke(:,col_theta));
    thetadispl(1) = [];
    %<VS
    
    % pairwise time diffs
    tdelta = filter([1 -1] ,1 ,x_stroke(:,col_time));
    tdelta(1) = [];
    
    
    % pairw angle
    angl = atan2(ydispl,xdispl);
    
    % Mean Resutlant Length (requires circular statistics toolbox)
    featMat(i, 10) = circ_r(angl) ;
    
    % pairwise displacements
    pairwDist = sqrt(xdispl.^2 + ydispl.^2);
    
    % speed histogram (Units of mm/sec)
    v = pairwDist./tdelta;
    featMat(i, 14:16) = prctile(v, indivPrctlVals);
    
    %>VS
    %Measurement of d(rho)/dt (20,50,80% change)
    drho = rhodispl./tdelta;
    featMat(i,39:41) = prctile(drho,indivPrctlVals);
    
    %Measurement of d(theta)/dt (20,50,80% change)
    dtheta = thetadispl./tdelta;
    featMat(i,42:44) = prctile(dtheta,indivPrctlVals);
    
    % Measurement of d^2(rho)/dt^2 (20,50,80% change)
    d2rho = filter([1 -1] ,1 ,drho);
    d2rho = d2rho./tdelta;
    d2rho(1) = [];
    featMat(i,45:47) = prctile(d2rho,indivPrctlVals);
    
    % Measurement of d^2(theta)/dt^2 (20,50,80% change)
    d2theta = filter([1 -1] ,1 ,dtheta);
    d2theta = d2theta./tdelta;
    d2theta(1) = [];
    featMat(i,48:50) = prctile(d2theta,indivPrctlVals);
    %<VS
    
    % acceleration histogram
    a = filter([1 -1] ,1 ,v);
    a = a./tdelta;
    a(1) = [];
    
    
    
    %% full stat stuff
    featMat(i, 17:19) = prctile(a, indivPrctlVals);
    
    
    %median velocity of last 3 points
    featMat(i, 20) = median(v(max([end-3 1]):end));
    
    %>VS
    %median drho/dt of last three points
    featMat(i, 51) = median(drho(max([end-3 1]):end));
 
    %median dtheta/dt of last three points
    featMat(i, 52) = median(dtheta(max([end-3 1]):end));
    
    %<VS
    
    % max dist. beween direct line and true line (with sign)
    xvek = x_stroke(:,col_x)-x_stroke(1,col_x);
    yvek = x_stroke(:,col_y)-x_stroke(1,col_y);
    
    % project each vector on straight line
    % compute unit line perpendicular to straight connection and project on
    % this
    perVek = cross([xvek(end) yvek(end) 0], [0 0 1]);
    perVek =  perVek / sqrt(([perVek(1) perVek(2)] * [perVek(1) perVek(2)]'));
    perVek(isnan(perVek)) = 0; % happens if vectors have lenght 0
    
    %all distances to direct line
    projectOnPerpStraight =...
        xvek  .* repmat(perVek(1) , [length(xvek) 1]) + ...
        yvek  .* repmat(perVek(2) , [length(xvek) 1]);
    
    % report maximal (absolute) distance
    absProj = abs(projectOnPerpStraight);
    maxind = find( absProj == max( absProj ) );
    featMat(i, 21) = projectOnPerpStraight(maxind(1));
    
    % stat of distances (bins are not the same for all strokes)
    featMat(i, 22:24) = prctile(projectOnPerpStraight, indivPrctlVals);
    
    % average direction of ensemble of pairs
    featMat(i, 25) = circ_mean(angl) ;
    
    %  direction of end-to-end line
    featMat(i, 12) = atan2( featMat(i, 8)-featMat(i, 6) , featMat(i, 7)-featMat(i, 5)) ;
    
    
    % direction flag 1 up, 2 down, 3 left 4 right  (see doc atan2): in what
    % direction is screen being moved to?
    tmpangle = featMat(i, 12) + pi; % convert to [0 2pi]
    if tmpangle <= pi/4
        featMat(i, 11) = 4; % right
    elseif tmpangle > pi/4 && tmpangle <= 5*pi/4 %  % up/left
        if tmpangle < 3*pi/4
            featMat(i, 11) = 1; % up
        else
            featMat(i, 11) = 2; % left
        end
    else % down / right
        if tmpangle < 7*pi/4
            featMat(i, 11) = 3; % down
        else
            featMat(i, 11) = 4; % right
        end
    end
    
    %length of trajectory
    featMat(i, 26) = sum(pairwDist);
    
    %ratio between direct length and length of trajectory
    featMat(i, 27) = featMat(i, 9) / featMat(i, 26);
        
    %average velocity
    featMat(i, 28) = featMat(i, 26) / featMat(i, 4);
    
    %average acc over first 5 points
    featMat(i, 29) = median(a(1: min([5 length(a)]) ));
        
    % pressure in the middle of the stroke
    featMat(i, 30) = median( x_stroke(floor(npoints/2):ceil(npoints/2) ,col_press) );
    
    % covered area in the middle of the stroke
    featMat(i, 31) = median( x_stroke(floor(npoints/2):ceil(npoints/2) ,col_area) );
    
    % finger orientation in the middle of the stroke
    featMat(i, 32) = median( x_stroke(floor(npoints/2):ceil(npoints/2) ,col_Forient) );
    
    % change of finger orientation during stroke
    featMat(i, 33) =  x_stroke( end ,col_Forient) - x_stroke( 1 ,col_Forient) ;
    
    % phone orientation
    featMat(i, 34) =  x_stroke( 1 ,col_orient) ;
    
end

% delete NaNs
featMat(isnan(featMat)) = 0;





%% how many strokes per doc and user? (output stats on screen)
users = unique( featMat(:,1) );
docs = unique( featMat(:,2) );
userStats.total = zeros(length(users), length(docs));
userStats.up = userStats.total;
userStats.down = userStats.total;
userStats.left = userStats.total;
userStats.right = userStats.total;

for d=1:length(docs)
    for u=1:length(users)
        userStats.total(u,d) = sum( single(featMat(featMat(:,2)==d,1)==u ) );
        userStats.up(u,d)    = sum( single(featMat(featMat(:,2)==d,1)==u ) .* single(featMat(featMat(:,2)==d, 11) == 1 )   );
        userStats.down(u,d)  = sum( single(featMat(featMat(:,2)==d,1)==u ) .* single(featMat(featMat(:,2)==d, 11) == 3 )   );
        userStats.left(u,d)  = sum( single(featMat(featMat(:,2)==d,1)==u ) .* single(featMat(featMat(:,2)==d, 11) == 2 )   );
        userStats.right(u,d) = sum( single(featMat(featMat(:,2)==d,1)==u ) .* single(featMat(featMat(:,2)==d, 11) == 4 )   );
    end
end
disp('stroke counts, nusers x ndocs' )
disp(num2str(userStats.total));
disp('up' )
disp(num2str(userStats.up));
disp('down' )
disp(num2str(userStats.down));
disp('left' )
disp(num2str(userStats.left));
disp('right' )
disp(num2str(userStats.right));


end

