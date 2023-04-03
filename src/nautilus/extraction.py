# feature_extraction.py
# Written by Thomas Hilder

# import packages
import numpy                as np
import matplotlib.pyplot    as plt

# import specific functions
from scipy.signal       import find_peaks, savgol_filter, peak_widths
from scipy.optimize     import curve_fit
from scipy.ndimage      import gaussian_filter1d
from sklearn.cluster    import DBSCAN
from scipy.optimize     import OptimizeWarning

# we want warnings to be errors
import warnings
warnings.filterwarnings("error")

def gaussian(x, a, x0, sigma):
    return a * np.exp(- (x - x0)**2 / (2 * sigma**2))

def fit_gaussian(x, y, yerr):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = curve_fit(
        gaussian, 
        x, 
        y,
        sigma = yerr,
        absolute_sigma=True,
        p0 = [max(y), mean, sigma],
        bounds = (
            [     0,      0,   1e-8],   # min bounds
            [np.inf, np.inf, np.inf]    # max bounds
        )
    )
    return popt, pcov
    
class Features():
    
    def __init__(self, data_polar, rad, phi, data_cartesian, x, y, remove_ends=False) -> None:
        """
        Initialise the features object.
        
        Parameters
        ----------
        data_polar : 2D array
            de-projected data on a polar grid
        rad : 1D array
            radii of de-projected polar data
        phi : 1D array
            azimuths of de-projected polar data
        data_cartesian : 2D array
            de-projected data on a cartsian grid
        x : 1D array
            x coords of de-projected cartsian data
        y : 1D array
            y coords of de-projected cartsian data
        remove_ends : bool
            whether or not to trim the azimuthal ends of the data due to de-projection effects. Default is False
        """
        
        # check inputs have the right shapes
        if data_polar.shape[0] != len(rad) or data_polar.shape[1] != len(phi):
            raise Exception("polar data must have shape (len(rad),len(phi))")
        if data_cartesian.shape[0] != len(x) or data_cartesian.shape[1] != len(y):
            raise Exception("cartesian data must have shape (len(x),len(y))")
        
        # remove the end-most azimuthal data if desired by the user
        if remove_ends:
            data = data[:,1:-1]
            phi  = phi[1:-1]
        
        # save polar data
        self.data_polar = np.copy(data_polar)
        self.rad        = np.copy(rad)
        self.phi        = np.copy(phi)
        
        # save cartesian data
        self.data_cartesian = np.copy(data_cartesian)
        self.x              = np.copy(x)
        self.y              = np.copy(y)
        
        # save useful constants
        self.N_RAD = len(rad)
        self.N_PHI = len(phi)
        self.N_X   = len(x)
        self.N_Y   = len(y)
    
    def extract_initial_points(self, sub_az_av=True, savgol=True, beam_size=1) -> None:
        """
        Extract all the points along bright arcs and spirals using scipy.signal.find_peaks.
        
        Parameters
        ----------
        apply_blur : bool
            Whether to apply a Gaussian blur in the azimuthal direction to remove channelisation effects
        blur_sigma_pix : float
            Width of the Gaussian blur in pixel
        sub_az_av : bool
            Whether to subtract an azimuthal average before applying find_peaks
        """
        
        # === Beam information === #
        
        # calculate radial pixel scale
        self.r_pixel_scale = self.rad.max() / self.N_RAD

        # calculate beam size in pixels
        beam_pix  = beam_size / self.r_pixel_scale
        print(f"beam size in pixels radially = {beam_pix:.3}")
        
        # save the beam sizes
        self.beam_arcsec = beam_size
        self.beam_pix    = beam_pix
        
        # === Prepare data === #
        
        # store data in local variable so we can apply changes
        data_for_extract = np.copy(self.data_polar)
        
        # # blur data in the azimuthal direction only
        # if apply_blur:
        #     data_for_extract = gaussian_filter1d(data_for_extract, sigma=blur_sigma_pix, axis=1)
            
        # subtract azimuthal average
        if sub_az_av:
            self.az_av = np.mean(data_for_extract, axis=1)
            data_for_extract = data_for_extract - self.az_av[:,np.newaxis]
            
        # apply savgol filter
        if savgol:
            data_for_extract = savgol_filter(data_for_extract, window_length=int(beam_pix), polyorder=3, axis=0)
        
        # save the data we used to extract points
        self.extract_data = data_for_extract
            
        # === Extract points === #
        
        # initialise list to store indices of extracted points
        indices = []

        # intialise lists to store extracted points
        rad_p = []
        phi_p = []

        # loop over all azimuths
        for index_phi in range(self.N_PHI):
            
            std = np.std(data_for_extract[:,index_phi])
        
            # extract indices of peaks
            indices_r = list(
                find_peaks(
                    data_for_extract[:,index_phi], width=beam_pix, height=0, prominence=std
                    )[0]
            )
            
            # loop over all extracted peaks
            for index_rad in indices_r:
                
                # append to indices
                indices.append((index_rad,index_phi))
                
                # append extracted points
                rad_p.append(self.rad[index_rad])
                phi_p.append(self.phi[index_phi])
                
        # === Save results === #
                
        # save indices
        self.peaks_indices = indices
        
        # convert results to arrays and save
        self.rad_p = np.array(rad_p)
        self.phi_p = np.array(phi_p)
    
    @staticmethod
    def get_cartesian_points(r_points, phi_points, flip_y=True, phi_offset=-90) -> None:
        # flip_y and phi_offset are to account in the difference in deprojection used in the Cartesian
        # and polar deprojections in Eddy
        
        if flip_y:
            fac = -1
        else:
            fac =  1
    
        x_keep =       r_points * np.cos((phi_points + phi_offset) * np.pi / 180)
        y_keep = fac * r_points * np.sin((phi_points + phi_offset) * np.pi / 180)
        
        return x_keep, y_keep
    
    def filter_points_derivatives(self) -> None:
        """
        Filter points by the second radial derivative. For each point
        
        Parameters
        ----------
        apply_blur : bool
            Whether to apply a Gaussian blur in the azimuthal direction to remove channelisation effects
        blur_sigma_pix : float
            Width of the Gaussian blur in pixel
        """
        
        # === Get derivatve information === #
        
        # store data in local variable so we can apply changes
        data_for_derivs = np.copy(self.data_polar)
        
        # # blur data in the azimuthal direction only
        # if apply_blur:
        #     data_for_derivs = gaussian_filter1d(data_for_derivs, sigma=blur_sigma_pix, axis=1)
            
        # calculate derivatives, keep second radial derivative
        d_dr  , _ = np.gradient(data_for_derivs, self.rad, self.phi)
        d2_dr2, _ = np.gradient(           d_dr, self.rad, self.phi)
        
        # === Get x, y coordinates for polar grid === #
        
        rad_mesh, phi_mesh = np.meshgrid(self.rad, self.phi)
        x_mesh, y_mesh = self.get_cartesian_points(rad_mesh, phi_mesh, flip_y=False, phi_offset=0)
        
        # === Filter points === #
        
        # initialise lists for results
        rad_keep = []
        phi_keep = []
        rad_rej  = []
        phi_rej  = []
        
        for i, pt in enumerate(self.peaks_indices):
            
            # get indices
            i_rad = pt[0]
            i_phi = pt[1]
            
            # get point
            rad_point = self.rad_p[i]
            phi_point = self.phi_p[i]
            x_point, y_point = self.get_cartesian_points(rad_point, phi_point, flip_y=False, phi_offset=0)
            
            # calculate 2D gaussian centred on point with sigma = beam / 2
            k = 0.5
            gaussian_window = self.gaussian_func_2D(
                x     = x_mesh, 
                y     = y_mesh, 
                x_0   = x_point, 
                y_0   = y_point,
                sig_x = self.beam_arcsec * k,
                sig_y = self.beam_arcsec * k
            )
            
            # sum weighted second radial derivative
            # plt.pcolormesh(self.phi, self.rad, gaussian_window.T * d2_dr2, cmap="RdBu", vmin=-0.5, vmax=0.5)
            # plt.scatter(phi_point, rad_point, c="k", marker="x", s=400)
            # plt.show()
            integ_d2_dr2 = np.sum(
                gaussian_window.T * d2_dr2
            )
            
            # if the result is negative, the point corresponds to crest and so we keep it   
            if integ_d2_dr2 < 0 :
                
                rad_keep.append(self.rad[i_rad])
                phi_keep.append(self.phi[i_phi])
                
            # otherwise reject the point
            else:
                rad_rej.append(self.rad[i_rad])
                phi_rej.append(self.phi[i_phi])
                
        # === Save results === #
                
        # update extracted points to the ones we kept
        self.rad_p = np.array(rad_keep)
        self.phi_p = np.array(phi_keep)
        
        # save rejected points
        self.rad_p_rejected = np.array(rad_rej)
        self.phi_p_rejected = np.array(phi_rej)
    
    @staticmethod
    def gaussian_func_2D(x, y, x_0=0, y_0=0, sig_x=1, sig_y=1, cut=2):
        #mask = np.sqrt((x - x_0)**2 + (y - y_0)**2) <= cut*np.sqrt(sig_x**2 + sig_y**2)
        result = (
            np.exp(
                -0.5 * (
                    (x - x_0)**2 / sig_x**2 + (y - y_0)**2 / sig_y**2
                )
            )
        )       
        return result
        
    def apply_mask(self, r_inner, r_outer) -> None:
        """
        Remove any extracted points interior to r_inner and exterior to r_outer. MUST be applied after filtering by derivatives if you wish to do that.
        
        Parameters
        ----------
        r_inner : float
            inner radius, points interior to r_inner are removed
        r_outer : float
            outer radius, points exterior to r_inner are removed
        """
        
        # copy points arrays
        phi_p = np.copy(self.phi_p)
        rad_p = np.copy(self.rad_p)
        try:
            u_rad_p = np.copy(self.u_rad_p)
            save_u  = True
        except:
            u_rad_p = np.zeros(self.rad_p.shape)
            save_u  = False
        
        # save unmasked points
        self.phi_p_unmasked = np.copy(phi_p)
        self.rad_p_unmasked = np.copy(rad_p)
        if save_u:
            self.u_rad_p_unmasked = np.copy(u_rad_p)
        
        # exclude points exterior to r_outer
        phi_p   = phi_p  [rad_p <= r_outer]
        u_rad_p = u_rad_p[rad_p <= r_outer]
        rad_p   = rad_p  [rad_p <= r_outer]
        
        # exclude points interior to r_inner
        phi_p   = phi_p  [rad_p >= r_inner]
        u_rad_p = u_rad_p[rad_p >= r_inner]
        rad_p   = rad_p  [rad_p >= r_inner]
        
        # save updated points
        self.phi_p = phi_p
        self.rad_p = rad_p
        if save_u:
            self.u_rad_p = u_rad_p
        
    def get_final_points_and_uncertainties_gaussian(self, rms, plots=True):
        
        # calculate uncertainty on az-av subtracted
        u_I = rms * np.sqrt(1 + 1 / self.N_PHI)
        print(u_I)
        
        # initialise array to store uncertainties
        self.u_rad_p = np.zeros((len(self.rad_p)))
        
        # point counter
        point_no = 0
        
        # loop over all azimuths
        for j, az in enumerate(self.phi):
            
            # find radial index of all points that are at this azimuth
            indices = []
            for i, p in enumerate(self.phi_p):
                if np.isclose(p, az):
                    for k, r in enumerate(self.rad):
                        if np.isclose(self.rad_p[i], r):
                            indices.append(k)
            
            # if peaks at this azimuth
            if len(indices) > 0:
                
                # grab points only in neighbourhood of peak according to ips values
                for ind in indices:
                    
                    # copy data arrays
                    r = np.copy(self.rad)
                    d = np.copy(self.extract_data[:,j])
                    
                    # restrict data to nearby point (within a beam)
                    fac = 1
                    d = d[r > self.rad_p[point_no] - fac * self.beam_arcsec]
                    r = r[r > self.rad_p[point_no] - fac * self.beam_arcsec]
                    d = d[r < self.rad_p[point_no] + fac * self.beam_arcsec]
                    r = r[r < self.rad_p[point_no] + fac * self.beam_arcsec]
                    
                    # also remove points below zero
                    r = r[d > 0]
                    d = d[d > 0]
                    
                    if plots:
                        plt.scatter(r, d)
                    
                    # check for bad fits and throw out those points by setting to zero
                    try:
                        popt, pcov = fit_gaussian(r, d, np.repeat(u_I, r.shape[0]))
                        #print(pcov)
                        perr = np.sqrt(np.diag(pcov))
                    except OptimizeWarning:
                        popt = np.array([0, 0, 1])
                        
                    # also check for points with huge errorbars
                    if perr[1] > 3.0 * self.beam_arcsec:
                        popt = np.array([0, 0, 1])
                    
                    if plots: 
                        plt.plot(self.rad, gaussian(self.rad, *popt))
                    
                    # update peak value
                    self.rad_p[point_no]   = popt[1] 
                    self.u_rad_p[point_no] = perr[1] #popt[2]
                    print(popt[1], perr[1])
                    point_no += 1
                    
                    # plot width
                    if plots:
                        #plt.hlines(0, popt[1]-popt[2], popt[1]+popt[2], colors="red", zorder=10)
                        plt.hlines(0, popt[1]-perr[1], popt[1]+perr[1], colors="red", zorder=10)
                if plots:
                    plt.errorbar(self.rad, self.extract_data[:,j], u_I, color="k", zorder=-1)
                    plt.show()
        
        # uncertainty in phi is half the space between phi points
        u_phi = 0.5 * 360 / self.N_PHI
        print(f"uncertainty in phi = {u_phi} degrees")
                    
                    
    def get_final_points_and_uncertainties_from_peak_width(self):
        
        # initialise array to store uncertainties
        self.u_rad_p = np.zeros((len(self.rad_p),2))
        
        # point counter
        point_no = 0
        
        # loop over all azimuths
        for j, az in enumerate(self.phi):
            
            # find radial index of all points that are at this azimuth
            indices = []
            for i, p in enumerate(self.phi_p):
                if np.isclose(p, az):
                    for k, r in enumerate(self.rad):
                        if np.isclose(self.rad_p[i], r):
                            indices.append(k)
            
            # if peaks at this azimuth
            if len(indices) > 0:
                
                # get width results
                widths = peak_widths(self.extract_data[:,j], indices, rel_height=0.3)
                
                # get intersections
                inner_ips, outer_ips = widths[2], widths[3]
                
                # convert intersections to arcseconds
                inner_ips *= self.r_pixel_scale
                outer_ips *= self.r_pixel_scale
                               
                #print(inner_ips, outer_ips)    
                plt.plot(self.rad, self.extract_data[:,j], color="k")
                plt.hlines(widths[1], inner_ips, outer_ips, color="red")
                plt.show()
                
                # save uncertainties
                for l in range(len(inner_ips)):
                    self.u_rad_p[point_no,0] = inner_ips[l]
                    self.u_rad_p[point_no,1] = outer_ips[l]
                    point_no += 1
    
    def find_clusters(self, eps=0.25, min_samples=5, plot_lim=2.5, **kwargs) -> None:
        """
        Use DBSCAN clustering algorithm to assign points to groups in order to convert collection of unassigned points into collection of features, each with their own points.
        
        Parameters
        ----------
        eps : float
            eps parameter passed to sklearn.cluster.DBSCAN
        min_samples : int
            min_samples parameter passed to sklearn.cluster.DBSCAN
        """
        
        # get cartesian coordinates of points
        x_p, y_p = self.get_cartesian_points(self.rad_p, self.phi_p, **kwargs)
        points = np.column_stack((x_p, y_p))
        
        # perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        
        self.labels = labels
        
        # print the number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Number of clusters: {n_clusters}")
        
        #create plot
        fig, ax = plt.subplots(figsize=[5,5], dpi=150)
        ax.set_title(f"DBSCAN Clustering (n_clusters={n_clusters})")
        ax.pcolormesh(self.x, self.y, self.data_cartesian, cmap="magma", edgecolors="face", rasterized=True)
        ax.scatter(x_p, y_p, c=labels, cmap='jet', s=4)
        ax.set_xlim( plot_lim, -plot_lim)
        ax.set_ylim(-plot_lim,  plot_lim)
        ax.set_xlabel(r'x ["]')
        ax.set_ylabel(r'y ["]')
        #plt.show()
        
        # initialise groups
        group_indices = [[] for _ in range(n_clusters)]
                
        # assign points to groups based on labels
        for i, label in enumerate(labels):
            
            # if label is -1 we just reject the point
            if label == -1:
                pass
            else:
                group_indices[label].append(i)
                
        # save results
        self.clustering_indices = group_indices