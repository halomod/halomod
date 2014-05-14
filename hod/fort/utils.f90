module utils
    contains

    subroutine simps(dx, func,val)
        implicit none
        !!simps works on functions defined at EQUIDISTANT values of x only

        real(8), intent(in) :: dx !The grid spacing
        real(8), intent(in) :: func(:) !The function
        real(8), intent(out) :: val !The value of the integral

        integer :: n !length of func
        integer :: i,start
        real(8) :: endbit

        n = size(func)
        if (mod(n,2)>0) then
            start = 2
            endbit = (func(1)+func(2))*dx/2
        else
            start = 1
            endbit=0.d0
        end if

        val = func(start) + func(n) + 4*func(n-1)
        do i=start,n/2-1
            val = val + 2*func(2*i)
            val = val + 4*func(2*i-1)
        end do
        val = val*dx/3 + endbit
    end subroutine

    subroutine spline (x, y, b, c, d, n)
    !======================================================================
    !  Calculate the coefficients b(i), c(i), and d(i), i=1,2,...,n
    !  for cubic spline interpolation
    !  s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
    !  for  x(i) <= x <= x(i+1)
    !  Alex G: January 2010
    !----------------------------------------------------------------------
    !  input..
    !  x = the arrays of data abscissas (in strictly increasing order)
    !  y = the arrays of data ordinates
    !  n = size of the arrays xi() and yi() (n>=2)
    !  output..
    !  b, c, d  = arrays of spline coefficients
    !  comments ...
    !  spline.f90 program is based on fortran version of program spline.f
    !  the accompanying function fspline can be used for interpolation
    !======================================================================
        implicit none

        integer, intent(in) :: n
        real(8), intent(in) :: x(n), y(n)
        real(8), intent(out):: b(n), c(n), d(n)

        integer :: i, j, gap
        real(8) :: h

        gap = n-1

        ! check input
        if ( n < 2 ) return
        if ( n < 3 ) then
          b(1) = (y(2)-y(1))/(x(2)-x(1))   ! linear interpolation
          c(1) = 0.
          d(1) = 0.
          b(2) = b(1)
          c(2) = 0.
          d(2) = 0.
          return
        end if
        !
        ! step 1: preparation
        !
        d(1) = x(2) - x(1)
        c(2) = (y(2) - y(1))/d(1)
        do i = 2, gap
          d(i) = x(i+1) - x(i)
          b(i) = 2.0*(d(i-1) + d(i))
          c(i+1) = (y(i+1) - y(i))/d(i)
          c(i) = c(i+1) - c(i)
        end do
        !
        ! step 2: end conditions
        !
        b(1) = -d(1)
        b(n) = -d(n-1)
        c(1) = 0.0
        c(n) = 0.0
        if(n /= 3) then
          c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
          c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
          c(1) = c(1)*d(1)**2/(x(4)-x(1))
          c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
        end if
        !
        ! step 3: forward elimination
        !
        do i = 2, n
          h = d(i-1)/b(i-1)
          b(i) = b(i) - h*d(i-1)
          c(i) = c(i) - h*c(i-1)
        end do
        !
        ! step 4: back substitution
        !
        c(n) = c(n)/b(n)
        do j = 1, gap
          i = n-j
          c(i) = (c(i) - d(i)*c(i+1))/b(i)
        end do
        !
        ! step 5: compute spline coefficients
        !
        b(n) = (y(n) - y(gap))/d(gap) + d(gap)*(c(gap) + 2.0*c(n))
        do i = 1, gap
          b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
          d(i) = (c(i+1) - c(i))/d(i)
          c(i) = 3.*c(i)
        end do
        c(n) = 3.0*c(n)
        d(n) = d(n-1)
    end subroutine spline

    function ispline(u, x, y, b, c, d, n)
    !======================================================================
    ! function ispline evaluates the cubic spline interpolation at point z
    ! ispline = y(i)+b(i)*(u-x(i))+c(i)*(u-x(i))**2+d(i)*(u-x(i))**3
    ! where  x(i) <= u <= x(i+1)
    !----------------------------------------------------------------------
    ! input..
    ! u       = the abscissa at which the spline is to be evaluated
    ! x, y    = the arrays of given data points
    ! b, c, d = arrays of spline coefficients computed by spline
    ! n       = the number of data points
    ! output:
    ! ispline = interpolated value at point u
    !=======================================================================
        implicit none
        real(8) :: ispline
        integer, intent(in) :: n
        real(8), intent(in) :: u, x(n), y(n), b(n), c(n), d(n)

        integer :: i, j, k
        real(8) :: dx

        ! if u is ouside the x() interval take a boundary value (left or right)
        if(u <= x(1)) then
          ispline = y(1)
          return
        end if
        if(u >= x(n)) then
          ispline = y(n)
          return
        end if

        !*
        !  binary search for for i, such that x(i) <= u <= x(i+1)
        !*
        i = 1
        j = n+1
        do while (j > i+1)
          k = (i+j)/2
          if(u < x(k)) then
            j=k
            else
            i=k
           end if
        end do
        !*
        !  evaluate spline interpolation
        !*
        dx = u - x(i)
        ispline = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))
    end function ispline


end module utils
