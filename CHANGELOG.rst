Changelog
=========

Version 0.1.5
-------------
- This updates Scipy for gaussian_filter (#20)
- Update thalamus (#18)
- Update condition for error from layer queries. Fix #16 (#17)

Version 0.1.4
-------------

- Added `direction-vectors from-center` command; a placeholder way to generate
  orientation vectors for a particular region, by supplying a central point,
  and emanating away

Version 0.1.2
-------------

- Make 2 direction builders available in the `atlas-direction-vectors direction-vectors` application:
    * layer-region: can specify for any layered region what the values are for
      certain layer voxels, such that the blur-gradient is more accurate

Version 0.1.1
-------------
- Open source release

Version 0.1.0
-------------
- Initial commit: extracts the `direction_vectors` from internal tools
