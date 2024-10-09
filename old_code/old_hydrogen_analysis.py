# plt.figure(figsize=(12, 5))
# plt.title('Distance in ')
# plt.xlabel('Dip indices between which the distance was measured')
# plt.ylabel('Distance in [resolution units]')
#
# for i, d_a in enumerate(diff_arrs):
#     # Scatter plot for data
#     plt.scatter(np.arange(len(d_a)), d_a, label=f'{file_indices[i]}')
#     plt.plot(np.arange(len(d_a)), d_a, ls='--', lw=0.75)
#
#     # Perform linear regression on the last three data points
#     x_last_three = np.arange(len(d_a))[-3:]
#     y_last_three = d_a[-3:]
#
#     # Linear fit (degree 1)
#     p, cov = np.polyfit(x_last_three, y_last_three, 1, cov=True)
#     slope, intercept = p
#     slope_err, intercept_err = np.sqrt(np.diag(cov))
#
#     # Generate x values for plotting the fit line
#     x_fit = np.linspace(x_last_three[0], x_last_three[-1], 100)
#     y_fit = slope * x_fit + intercept
#
#     # Calculate 1-sigma confidence bands
#     y_fit_upper = (slope + slope_err) * x_fit + (intercept + intercept_err)
#     y_fit_lower = (slope - slope_err) * x_fit + (intercept - intercept_err)
#
#     # Plot the regression line
#     # plt.plot(x_fit, y_fit, label=f'Fit {file_indices[i]}')
#
#     # Plot the 1-sigma confidence band
#     # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label=f'1-sigma band {file_indices[i]}')
#
# plt.legend()
# plt.xlim(-0.25, 2.25)
# plt.xticks([0, 1, 2], [r'$1 \rightarrow 2$', r'$2 \rightarrow 3$', r'$3 \rightarrow 4$'])
# plt.tight_layout()
# plt.show()